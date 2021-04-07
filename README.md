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

## Introduction for ViSha
ViSha is a short name for "Video shadow detection dataset". As mentioned in above paper, our dataset includes 120 videos with diverse content, varying length, and object-level annotations. More than half videos are from 5 widely-used video tracking benchmarks (i.e., OTB, VOT, LaSOT, TC-128, and NfS). The remaining 59 videos are self-captured with different hand-held cameras, over different scenes, at varying times. The frame rate is adjusted to 30 fps for all video sequences. Eventually, our video shadow detection dataset (ViSha) contains 120 video sequences, with a totally of 11,685 frames and 390 seconds duration. The longest video contains 103 frames and the shortest contains 11 frames.

To provide guidelines for future works, we randomly split the dataset into training and testing sets with a ratio of 5:7. The 50 training set and 70 testing set can be available in [[BaiduNetdisk](https://pan.baidu.com/s/1DYjXERQuIlbtNPe4wFcJXA)](pw: q0lh) or [Google Drive()].

If you download ViSha and unzip each file, you can find the dataset structure as follows:

ViSha_release  
+-- train  
|&nbsp;&nbsp;&nbsp;+-- images
|   |   +-- baby_cat  
|   |   |   +-- 00000001.jpg  
|   |   |   +-- 00000002.jpg  
|   |   |   +-- ...  
|   |   +-- baby_wave1 
|   |   |   +-- 00000001.jpg  
|   |   |   +-- 00000002.jpg  
|   |   |   +-- ...  
|   |   +-- ...  
|   +-- labels
|   |   +-- baby_cat  
|   |   |   +-- 00000001.png  
|   |   |   +-- 00000002.png  
|   |   |   +-- ...  
|   |   +-- baby_wave1 
|   |   |   +-- 00000001.png  
|   |   |   +-- 00000002.png  
|   |   |   +-- ...  
|   |   +-- ...  
+-- test  
|   +-- ...



## Trained Model
You can download the trained model which is reported in our paper at [BaiduNetdisk](https://pan.baidu.com/s/1yjnsjE7mDPnEaHxdtNFhhQ)(password: h52i).

## Requirement
* Python 3.6
* PyTorch 1.3.1(After 0.4.0 would be ok)
* torchvision
* numpy
* tqdm
* PIL
* pydensecrf ([here](https://github.com/Andrew-Qibin/dss_crf) to install)

## Training
1. Set ...
2. Set ...
3. Run by ```python train.py```

The pretrained ResNeXt model is ported from the [official](https://github.com/facebookresearch/ResNeXt) torch version,
using the [convertor](https://github.com/clcarwin/convert_torch_to_pytorch) provided by clcarwin. 
You can directly [download](https://drive.google.com/open?id=1dnH-IHwmu9xFPlyndqI6MfF4LvH6JKNQ) the pretrained model ported by me.

## Testing
1. Set ...
2. Put ...
2. Run by ```python infer.py```

## Useful links
UCF dataset: [Google Drive](https://drive.google.com/open?id=12DOmMVmE-oNuJVXmkBJrkfBvuDd0O70N) or [BaiduNetdisk](https://pan.baidu.com/s/1zt9ya1lzNcoGoc2CET3mdg)(password:o4ub for BaiduNetdisk)

SBU dataset: [BaiduNetdisk](https://pan.baidu.com/s/1FYQYLSkuTivjaRJVjjJhJw)(password:38qw for BaiduNetdisk)

Part of unlabel data that collected from internet: [BaiduNetdisk](https://pan.baidu.com/s/1_kdpwBlZ-K6gcZz45Tcg7g)(password: n1nb for BaiduNetdisk)
