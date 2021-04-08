# TVSD & ViSha
Code and dataset for the CVPR 2021 paper "Triple-cooperative Video Shadow Detection"

by Zhihao Chen, Liang Wan, Lei Zhu, Jia Shen, Huazhu Fu, Wennan Liu, and Jing Qin [[arXiv link](https://arxiv.org/abs/2103.06533)]

#### News: In 2021.4.7, We first release the code of TVSD and ViSha dataset.

***

## Citation
@inproceedings{chen21TVSD,   
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Chen, Zhihao and Wan, Liang and Zhu, Lei and Shen, Jia and Fu, Huazhu and Liu, Wennan and Qin, Jing},    
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Triple-cooperative Video Shadow Detection},    
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle = {CVPR},    
&nbsp;&nbsp;&nbsp;&nbsp;  year  = {2021}    
}

## Dataset
ViSha dataset is available at **[ViSha Homepage](https://erasernut.github.io/)**

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
2. Make sure you have a snapshot in ./models/TVSD (Tips: You can download the trained model which is reported in our paper at [BaiduNetdisk](https://pan.baidu.com/s/17d-wLwA5oyafMdooJlesyw)(pw: 8p5h) or [Google Drive])
4. Run by ```python infer.py``` to generate predicted masks
5. Run by ```python evaluate.py``` to evaluate above results

## Results in ViSha testing set
In ViSha testing set, we evaluate 12 related methods as follows: BDRAR[1], DSD[2], MTMT[3], FPN[4], PSPNet[5], DSS[6], R^3 Net[7], PDBM[8], COSNet[9], MAG[10], FEELVOS[11], STM[12]  
<small>[1]L. Zhu, Z. Deng, X. Hu, C.-W. Fu, X. Xu, J. Qin, and P.-A. Heng. Bidirectional feature pyramid network with recurrent attention residual modules for shadow detection. In ECCV, pages 121–136, 2018.  
[2]Q. Zheng, X. Qiao, Y. Cao, and R.W. Lau. Distraction-aware shadow detection. In CVPR, pages 5167–5176, 2019.  
[3]Z. Chen, L. Zhu, L. Wan, S. Wang, W. Feng, and P.-A. Heng. A multi-task mean teacher for semi-supervised shadow detection. In CVPR, pages 5611–5620, 2020.  
[4]T.-Y. Lin, P. Doll´ar, R. Girshick, K. He, B. Hariharan, and S.Belongie. Feature pyramid networks for object detection. In CVPR, pages 2117–2125, 2017.  
[5]H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia. Pyramid scene parsing network. In CVPR, pages 2881–2890, 2017.  
[6]Q. Hou, M. Cheng, X. Hu, A. Borji, Z. Tu, and P. Torr. Deeply supervised salient object detection with short connections. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(4):815–828, 2019.  
[7]Z. Deng, X. Hu, L. Zhu, X. Xu, J. Qin, G. Han, and P.-A. Heng. R3net: Recurrent residual reﬁnement network for saliency detection. In IJCAI, pages 684–690. AAAI Press, 2018.  
[8]H. Song, W. Wang, S. Zhao, J. Shen, and K.-M. Lam. Pyramid dilated deeper convlstm for video salient object detection. In ECCV, pages 715–731, 2018.  
[9]X. Lu, W. Wang, C. Ma, J. Shen, L. Shao, and F. Porikli. See more, know more: Unsupervised video object segmentation with co-attention siamese networks. In CVPR, pages 3623–3632, 2019.  
[10]H. Li, G. Chen, G. Li, and Y. Yu. Motion guided attention for video salient object detection. In ICCV, pages 7274–7283, 2019.  
[11]P. Voigtlaender, Y. Chai, F. Schroff, H. Adam, B. Leibe, and L.-C. Chen. Feelvos: Fast end-to-end embedding learning for video object segmentation. In CVPR, June 2019.  
[12]S.W. Oh, J.-Y. Lee, N. Xu, and S.J. Kim. Video object segmentation using space-time memory networks. In ICCV, pages 9226–9235, 2019.</small>

You can obtain those results and the results of our TVSD in [BaiduNetdisk](https://pan.baidu.com/s/1t_PgW3JCrTGvf_PVyeR-iw)(pw: ritw) or [Google Drive]
