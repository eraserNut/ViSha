import numpy as np
import os

import torch
from PIL import Image
from torchvision import transforms

from config import ViSha_validation_root
from misc import check_mkdir
from networks.TVSD import TVSD
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--snapshot', type=str, default='2', help='snapshot')
parser.add_argument('--models', type=str, default='TVSD', help='model name')
tmp_args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ckpt_path = './models'
exp_name = tmp_args.models
args = {
    'snapshot': tmp_args.snapshot,
    'scale': 416,
    'test_adjacent': 5,
    'input_folder': 'images',
    'label_folder': 'labels'
}

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

root = ViSha_validation_root[0]

to_pil = transforms.ToPILImage()


def main():
    net = TVSD().cuda()

    if len(args['snapshot']) > 0:
        check_point = torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'))
        net.load_state_dict(check_point['model'])

    net.eval()
    with torch.no_grad():
        video_list = os.listdir(os.path.join(root, args['input_folder']))
        for video in tqdm(video_list):
            # all images
            img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, args['input_folder'], video)) if
                        f.endswith('.jpg')]
            # need evaluation images
            img_eval_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, args['label_folder'], video)) if
                        f.endswith('.png')]

            img_eval_list = sortImg(img_eval_list)
            for exemplar_idx, exemplar_name in enumerate(img_eval_list):
                query_idx_list = getAdjacentIndex(exemplar_idx, 0, len(img_list), args['test_adjacent'])
                for query_idx in query_idx_list:
                    exemplar = Image.open(os.path.join(root, args['input_folder'], video, exemplar_name + '.jpg')).convert('RGB')
                    w, h = exemplar.size
                    query = Image.open(os.path.join(root, args['input_folder'], video, img_list[query_idx] + '.jpg')).convert('RGB')
                    exemplar_tensor = img_transform(exemplar).unsqueeze(0).cuda()
                    query_tensor = img_transform(query).unsqueeze(0).cuda()
                    exemplar_pre, _, _, _ = net(exemplar_tensor, query_tensor, query_tensor)
                    res = (exemplar_pre.data > 0).to(torch.float32)
                    prediction = np.array(
                        transforms.Resize((h, w))(to_pil(res.squeeze(0).cpu())))
                    check_mkdir(os.path.join(ckpt_path, exp_name, "predict_" + args['snapshot'], video))
                    # save form as 00000001_1.png, 000000001_2.png
                    save_name = f"{exemplar_name}_by{query_idx}.png"
                    print(os.path.join(ckpt_path, exp_name, "predict_" + args['snapshot'], video, save_name))
                    Image.fromarray(prediction).save(
                        os.path.join(ckpt_path, exp_name, "predict_" + args['snapshot'], video, save_name))


def sortImg(img_list):
    img_int_list = [int(f) for f in img_list]
    sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
    return [img_list[i] for i in sort_index]


def getAdjacentIndex(current_index, start_index, video_length, adjacent_length):
    if current_index + adjacent_length < start_index + video_length:
        query_index_list = [current_index+i+1 for i in range(adjacent_length)]
    else:
        query_index_list = [current_index-i-1 for i in range(adjacent_length)]
    return query_index_list

if __name__ == '__main__':
    main()
