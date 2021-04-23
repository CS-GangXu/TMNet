<p align="center">
  <img src="./tmnet_logo.png" width="175">
</p>

This is the official PyTorch implementation of TMNet in the CVPR 2021 paper "Temporal Modulation Network for Controllable Space-Time VideoSuper-Resolution"[[PDF]](https://arxiv.org/pdf/2104.10642v1.pdf). Our TMNet can flexibly interpolate intermediate frames for space-time video super-resolution (STVSR). 

## Contents
0. [Requirements](#Requirements)
0. [Installation](#Installation)
0. [Demo](#Demo)
0. [Training](#Training)
0. [Testing](#Testing)
0. [Citations](#Citations)

## Requirements
- Python 3.6
- PyTorch >= 1.1
- NVIDIA GPU + CUDA
- [Deformable Convolution v2](https://arxiv.org/abs/1811.11168), we adopt [CharlesShang's implementation](https://github.com/CharlesShang/DCNv2) in the submodule.
- Python packages: ```pip install numpy opencv-python lmdb pyyaml pickle5 matplotlib seaborn```

## Installation
First, make sure your machine has a GPU, which is required for the DCNv2 module.

1. Clone the TMNet repository.
```Shell
git clone --recursive https://github.com/CS-GangXu/TMNet.git
```
2. Compile the DCNv2:
```Shell
cd $ROOT/codes/models/modules/DCNv2
bash make.sh
python test.py
```
## Demo (To be uploaded at April 24, 2021 11:59PM (Pacific Time))

## Training (To be uploaded at April 24, 2021 11:59PM (Pacific Time))

## Testing (To be uploaded at April 24, 2021 11:59PM (Pacific Time))

## Citations

If you find the code helpful in your research or work, please cite the following papers.

```BibTeX
@InProceedings{xu2021temporal,
  author = {Gang Xu and Jun Xu and Zhen Li and Liang Wang and Xing Sun and Mingming Cheng},
  title = {Temporal Modulation Network for Controllable Space-Time VideoSuper-Resolution},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2021}
}

@InProceedings{xiang2020zooming,
  author = {Xiang, Xiaoyu and Tian, Yapeng and Zhang, Yulun and Fu, Yun and Allebach, Jan P. and Xu, Chenliang},
  title = {Zooming Slow-Mo: Fast and Accurate One-Stage Space-Time Video Super-Resolution},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={3370--3379},
  month = {June},
  year = {2020}
}

@InProceedings{wang2019edvr,
  author    = {Wang, Xintao and Chan, Kelvin C.K. and Yu, Ke and Dong, Chao and Loy, Chen Change},
  title     = {EDVR: Video restoration with enhanced deformable convolutional networks},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  month     = {June},
  year      = {2019},
}
```

## Acknowledgments
Our code is inspired by [Zooming-Slow-Mo-CVPR-2020](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020) and [EDVR](https://github.com/xinntao/EDVR).

## Contact
If you have any questions, feel free to E-mail Gang Xu with gangxu@mail.nankai.edu.cn.
