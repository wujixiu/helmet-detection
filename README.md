# Helmet Detection on Construction Sites Using SSD Framework

## Introduction
This project uses the SSD, one of most famous object detection frameworks, to detect whether a peron wearing the helmet 
and what color the helmet is on construction sites. So far, it's able to detect 4 kind of colors, including red, blue, white
and yellow. Besides, two different kind of models are fine-tuned for detection:

* SSD + VGG16 (original implemention, high accuracy but low efficiency)
* SSD + [Pelee](https://github.com/Robert-JunWang/Pelee) (for more detail, see the ICLR2018 workshop [paper](https://arxiv.org/pdf/1804.06882.pdf))

**note: For some reasons, this project only provides the model for testing, and in the future we will release the training data.**

## Preparation
1.Install SSD (https://github.com/weiliu89/caffe/tree/ssd) following the instructions there.

2.Clone this repository

```shell
git clone https://github.com/wujixiu/helmet-detection.git
```
3.Download the models from Google Drive or Baidu Yun and put them in models/vgg or models/pelee respectively.

## Testing
```shell
cd helmet-detection
python ssd_vgg.py
```

## Results
![results](test_imgs/001_results.jpg)
![results](test_imgs/003_results.jpg)
![results](test_imgs/005_results.jpg)
![results](test_imgs/004_results.jpg)
![results](test_imgs/002_results.jpg)

## Models
* VGG16 (97.6M)
* Pelee (20.3M)

[Google Drive](https://drive.google.com/file/d/1BANylV0M8IUTWvN_vX2BEMgx22Erx6ii/view?usp=sharing)

[Baidu Yun](https://pan.baidu.com/s/1-lFDPdhWF8haUwxXfynX7Q) (pwd:z4k9)






















