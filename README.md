# DFL-CNN : a fine-grained classifier
This is a simple pytorch re-implementation of CVPR 2018 [Learning a Discriminative Filter Bank Within a CNN for Fine-Grained Recognition](https://arxiv.org/pdf/1611.09932.pdf).

### Introduction:
This work still need to be updated.
The features are summarized blow:
+ Use VGG16 as base Network.
+ Dataset [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), you can split **trainset/testset** by yourself.**Or** you can download dataset which has been split directly from [BaiduYun Link](https://pan.baidu.com/s/1JQxa3DYDrM329skC73kbzQ).
+ This work has been trained on 4 Titan V after epoch 120 with batchsize 56, Now I got best result **Top1 85.140% Top5 96.237%** which is lower than author's. You can download weights from [weights](https://pan.baidu.com/s/1nxI3mV2cOOoMCLpCqg_cjA).
+ Part FCs is replaced by Global Average Pooling to reduce parameters.
+ Every some epoches, ten best patches is visualized in **vis_result** directory, you can put images you want to visualize in **vis_img** named number.jpg.
+ Update: ResNet-101 DFL-CNN and Multi-scale DFL-CNN need to be done.



### Algorithms Introduction:
![Display](https://github.com/songdejia/DFL-CNN/blob/master/screenshot/introduction2.png)

![Display](https://github.com/songdejia/DFL-CNN/blob/master/screenshot/introduction1.jpg)
 
 
 
### Results and Visualization of ten boxes for discriminative patches:
+ This work has been trained on 4 Titan V after epoch 120 with batchsize 56, Now I got best result **Top1 85.140% Top5 96.237%** which is lower than author's. You can download weights from [weights](https://pan.baidu.com/s/1nxI3mV2cOOoMCLpCqg_cjA). If use TenCrop transform in code, result can improve further.

+ Test Results:
<div align=center><img width="560" height="720" src="https://github.com/songdejia/DFL-CNN/blob/master/screenshot/test.jpg"/></div>

+ Visualization:
<div align=center><img width="700" height="700" src="https://github.com/songdejia/DFL-CNN/blob/master/screenshot/vis_1.jpg"/></div>

<div align=center><img width="700" height="700" src="https://github.com/songdejia/DFL-CNN/blob/master/screenshot/vis_2.jpg"/></div>

<div align=center><img width="700" height="700" src="https://github.com/songdejia/DFL-CNN/blob/master/screenshot/vis_3.jpg"/></div>

<div align=center><img width="700" height="700" src="https://github.com/songdejia/DFL-CNN/blob/master/screenshot/vis_4.jpg"/></div>




### Usage:
+ Download dataset, you can split trainset/valset by yourself
```
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
```
+ Or you can directly get it from [BaiduYun Link](https://pan.baidu.com/s/1JQxa3DYDrM329skC73kbzQ)
+ Then link original dataset to our code root/dataset
``` 
 ln -s ./train path/to/code/dataset/train 
 ln -s ./test  path/to/code/dataset/test
```
+ Finally, Train and Test.
+ Check you GPU resources and modify your run.sh. 
```
sh run.sh
```


### Note:
1. Visualization of ten best boxes is saved in **vis_result/**, img you want to visualize should be put
   in **vis_img/**. 
2. Weight(checkpoint.pth.tar, model_best.pth.tar) is in **weight/**.
3. Loss info is saved in **log/**.

