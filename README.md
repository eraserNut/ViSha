# TVSD & ViSha
Code and dataset for the CVPR 2021 paper "Triple-cooperative Video Shadow Detection"

by Zhihao Chen, Liang Wan, Lei Zhu, Jia Shen, Huazhu Fu, Wennan Liu, and Jing Qin [[arXiv link](https://arxiv.org/abs/2103.06533)]

#### News: In 2021.4.7, We first release the code of TVSD and ViSha dataset.

***

## Citation
@inproceedings{chen21TVSD,   
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Chen, Zhihao and Wan, Liang and Zhu, Lei and Shen, Jia and Fu, Huazhu and Liu, Wennan and Qin, Jing},    
&nbsp;&nbsp;&nbsp;&nbsp;  title = {A Multi-task Mean Teacher for Semi-supervised Shadow Detection},    
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle = {CVPR},    
&nbsp;&nbsp;&nbsp;&nbsp;  year  = {2021}    
}

## Dataset
ViSha dataset is available at [ViSha Homepage](https://erasernut.github.io/)

## Requirement
* Python 3.6
* PyTorch 1.3.1
* torchvision
* numpy
* tqdm
* PIL
* math
* time
* datatime
* argparse
* apex (alternative, fp16 for save memory and speedup)

## Training
1. Modify the data path on ./config.py
2. Modify the pretrained backbone path on ./networks/resnext_modify/config.py
3. Run by ```python train.py``` and model will be saved in ./models/TVSD

The pretrained ResNeXt model is ported from the [official](https://github.com/facebookresearch/ResNeXt) torch version,
using the [convertor](https://github.com/clcarwin/convert_torch_to_pytorch) provided by clcarwin. 
You can directly [download](https://drive.google.com/open?id=1dnH-IHwmu9xFPlyndqI6MfF4LvH6JKNQ) the pretrained model ported by me.

## Testing
1. Modify the data path on ./config.py
2. Make sure you have a snapshot in ./models/TVSD (Tips: You can download the trained model which is reported in our paper at [BaiduNetdisk](https://pan.baidu.com/s/17d-wLwA5oyafMdooJlesyw)(pw: 8p5h).)
4. Run by ```python infer.py``` to generate predicted masks
5. Run by ```python evaluate.py``` to evaluate above results

## Results in ViSha testing set
In ViSha testing set, we evaluate 12 related methods as follows:BDRAR[1], DSD[2], 
You can obtain it by [BaiduNetdisk]()
