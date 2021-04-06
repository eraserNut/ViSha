import numpy as np
import os
from PIL import Image
from misc import check_mkdir, cal_precision_recall_mae, AvgMeter, cal_fmeasure, cal_Jaccard, cal_BER
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, default='TVSD', help='model name')
parser.add_argument('--snapshot', type=str, default='12', help='model name')
tmp_args = parser.parse_args()

root_path = f'/home/ext/chenzhihao/code/video_shadow/models/{tmp_args.models}/predict_{tmp_args.snapshot}'
save_path = f'/home/ext/chenzhihao/code/video_shadow/models/{tmp_args.models}/predict_fuse_{tmp_args.snapshot}'

gt_path = '/home/ext/chenzhihao/Datasets/ViSha/test/labels'
input_path = '/home/ext/chenzhihao/Datasets/ViSha/test/images'

precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
mae_record = AvgMeter()
Jaccard_record = AvgMeter()
BER_record = AvgMeter()
shadow_BER_record = AvgMeter()
non_shadow_BER_record = AvgMeter()

video_list = os.listdir(root_path)
for video in tqdm(video_list):
    gt_list = os.listdir(os.path.join(gt_path, video))
    img_list = [f for f in os.listdir(os.path.join(root_path, video)) if f.split('_', 1)[0]+'.png' in gt_list]  # include overlap images
    img_set = list(set([img.split('_', 1)[0] for img in img_list]))  # remove repeat
    for img_prefix in img_set:
        # jump exist images
        check_mkdir(os.path.join(save_path, video))
        save_name = os.path.join(save_path, video, '{}.png'.format(img_prefix))
        # if not os.path.exists(os.path.join(save_path, video, save_name)):
        imgs = [img for img in img_list if img.split('_', 1)[0] == img_prefix]  # imgs waited for fuse
        fuse = []
        for img_path in imgs:
            img = np.array(Image.open(os.path.join(root_path, video, img_path)).convert('L')).astype(np.float32)
            # if np.max(img) > 0:  # normalize prediction mask
            #     img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            fuse.append(img)
        fuse = (sum(fuse) / len(imgs)).astype(np.uint8)
        # save image
        print(f'Save:{save_name}')
        Image.fromarray(fuse).save(save_name)
        # else:
        #     print(f'Exist:{save_name}')
        #     fuse = np.array(Image.open(save_name).convert('L')).astype(np.uint8)
        # calculate metric
        gt = np.array(Image.open(os.path.join(gt_path, video, img_prefix+'.png')))
        precision, recall, mae = cal_precision_recall_mae(fuse, gt)
        Jaccard = cal_Jaccard(fuse, gt)
        Jaccard_record.update(Jaccard)
        BER, shadow_BER, non_shadow_BER = cal_BER(fuse, gt)
        BER_record.update(BER)
        shadow_BER_record.update(shadow_BER)
        non_shadow_BER_record.update(non_shadow_BER)
        for pidx, pdata in enumerate(zip(precision, recall)):
            p, r = pdata
            precision_record[pidx].update(p)
            recall_record[pidx].update(r)
        mae_record.update(mae)

fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                        [rrecord.avg for rrecord in recall_record])
log = 'MAE:{}, F-beta:{}, Jaccard:{}, BER:{}, SBER:{}, non-SBER:{}'.format(mae_record.avg, fmeasure, Jaccard_record.avg, BER_record.avg, shadow_BER_record.avg, non_shadow_BER_record.avg)
print(log)


