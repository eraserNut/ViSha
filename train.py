import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import joint_transforms
from config import ViSha_training_root
from dataset.VShadow_crosspairwise import CrossPairwiseImg
from misc import AvgMeter, check_mkdir
from networks.TVSD import TVSD
from torch.optim.lr_scheduler import StepLR
import math
from losses import lovasz_hinge, binary_xloss
import random
import torch.nn.functional as F
import numpy as np
from apex import amp
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.deterministic = True
cudnn.benchmark = False

ckpt_path = './models'
exp_name = 'TVSD'

args = {
    'max_epoch': 12,
    'train_batch_size': 5,
    'last_iter': 0,
    'finetune_lr': 5e-5,
    'scratch_lr': 5e-4,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 416,
    'multi-scale': None,
    'gpu': '0,1',
    'multi-GPUs': False,
    'fp16': True,
    'warm_up_epochs': 3,
    'seed': 2020
}
# fix random seed
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])

# multi-GPUs training
if args['multi-GPUs']:
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    batch_size = args['train_batch_size'] * len(args['gpu'].split(','))
# single-GPU training
else:
    torch.cuda.set_device(0)
    batch_size = args['train_batch_size']

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale'], args['scale']))
])
val_joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

print('=====>Dataset loading<======')
training_root = [ViSha_training_root] # training_root should be a list form, like [datasetA, datasetB, datasetC], here we use only one dataset.
train_set = CrossPairwiseImg(training_root, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, shuffle=True)
print("max epoch:{}".format(args['max_epoch']))

ce_loss = nn.CrossEntropyLoss()

log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')

def main():
    print('=====>Prepare Network {}<======'.format(exp_name))
    # multi-GPUs training
    if args['multi-GPUs']:
        net = torch.nn.DataParallel(TVSD()).cuda().train()
        params = [
            {"params": net.module.encoder.parameters(), "lr": args['finetune_lr']},
            {"params": net.module.co_attention.parameters(), "lr": args['scratch_lr']},
            {"params": net.module.encoder.final_pre.parameters(), "lr": args['scratch_lr']},
            {"params": net.module.co_attention.parameters(), "lr": args['scratch_lr']},
            {"params": net.module.project.parameters(), "lr": args['scratch_lr']},
            {"params": net.module.final_pre.parameters(), "lr": args['scratch_lr']}
        ]
    # single-GPU training
    else:
        net = TVSD().cuda().train()
        params = [
            {"params": net.encoder.backbone.parameters(), "lr": args['finetune_lr']},
            {"params": net.encoder.aspp.parameters(), "lr": args['scratch_lr']},
            {"params": net.encoder.final_pre.parameters(), "lr": args['scratch_lr']},
            {"params": net.co_attention.parameters(), "lr": args['scratch_lr']},
            {"params": net.project.parameters(), "lr": args['scratch_lr']},
            {"params": net.final_pre.parameters(), "lr": args['scratch_lr']}
        ]

    # optimizer = optim.SGD(params, momentum=args['momentum'], weight_decay=args['weight_decay'])
    optimizer = optim.Adam(params, betas=(0.9, 0.99), eps=6e-8, weight_decay=args['weight_decay'])
    warm_up_with_cosine_lr = lambda epoch: epoch / args['warm_up_epochs'] if epoch <= args['warm_up_epochs'] else 0.5 * \
                             (math.cos((epoch-args['warm_up_epochs'])/(args['max_epoch']-args['warm_up_epochs'])*math.pi)+1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # change learning rate after 20000 iters

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    if args['fp16']:
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    train(net, optimizer, scheduler)


def train(net, optimizer, scheduler):
    curr_epoch = 1
    curr_iter = 1
    start = 0
    print('=====>Start training<======')
    while True:
        loss_record1, loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss_record5, loss_record6, loss_record7, loss_record8 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        for i, sample in enumerate(tqdm(train_loader, desc=f'Epoch: {curr_epoch}', ncols=100, ascii=' =', bar_format='{l_bar}{bar}|')):

            exemplar, exemplar_gt, query, query_gt = sample['exemplar'].cuda(), sample['exemplar_gt'].cuda(), sample['query'].cuda(), sample['query_gt'].cuda()
            other, other_gt = sample['other'].cuda(), sample['other_gt'].cuda()

            optimizer.zero_grad()

            exemplar_pre, query_pre, other_pre, scene_logits = net(exemplar, query, other)

            bce_loss1 = binary_xloss(exemplar_pre, exemplar_gt)
            bce_loss2 = binary_xloss(query_pre, query_gt)
            bce_loss3 = binary_xloss(other_pre, other_gt)

            loss_hinge1 = lovasz_hinge(exemplar_pre, exemplar_gt)
            loss_hinge2 = lovasz_hinge(query_pre, query_gt)
            loss_hinge3 = lovasz_hinge(other_pre, other_gt)

            loss_seg = bce_loss1 + bce_loss2 + bce_loss3 + loss_hinge1 + loss_hinge2 + loss_hinge3
            # classification loss
            scene_labels = torch.zeros(scene_logits.shape[0], dtype=torch.long).cuda()
            cla_loss = ce_loss(scene_logits, scene_labels) * 10
            loss = loss_seg + cla_loss

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 12)  # gradient clip
            optimizer.step()  # change gradient

            loss_record1.update(bce_loss1.item(), batch_size)
            loss_record2.update(bce_loss2.item(), batch_size)
            loss_record3.update(bce_loss3.item(), batch_size)
            loss_record4.update(loss_hinge1.item(), batch_size)
            loss_record5.update(loss_hinge2.item(), batch_size)
            loss_record6.update(loss_hinge3.item(), batch_size)
            loss_record7.update(cla_loss.item(), batch_size)

            curr_iter += 1

            log = "iter: %d, bce1: %f5, bce2: %f5, bce3: %f5, hinge1: %f5, hinge2: %f5, hinge3: %f5, cla: %f5, lr: %f8"%\
                  (curr_iter, loss_record1.avg, loss_record2.avg, loss_record3.avg, loss_record4.avg, loss_record5.avg,
                   loss_record6.avg, loss_record7.avg, scheduler.get_lr()[0])

            if (curr_iter-1) % 20 == 0:
                elapsed = (time.clock() - start)
                start = time.clock()
                log_time = log + ' [time {}]'.format(elapsed)
                print(log_time)
            open(log_path, 'a').write(log + '\n')

        if curr_epoch % 1 == 0:
            if args['multi-GPUs']:
                # torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_epoch))
                checkpoint = {
                    'model': net.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict()
                }
                torch.save(checkpoint, os.path.join(ckpt_path, exp_name, f'{curr_epoch}.pth'))
            else:
                # torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_epoch))
                checkpoint = {
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict()
                }
                torch.save(checkpoint, os.path.join(ckpt_path, exp_name, f'{curr_epoch}.pth'))
        if curr_epoch > args['max_epoch']:
            # torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
            return

        curr_epoch += 1
        scheduler.step()  # change learning rate after epoch


if __name__ == '__main__':
    main()