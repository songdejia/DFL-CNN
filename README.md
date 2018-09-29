# DFL-CNN : a fine-grained classifier
This is a simple pytorch re-implementation of [Learning a Discriminative Filter Bank Within a CNN for Fine-Grained Recognition](https://arxiv.org/pdf/1611.09932.pdf).


### 1.Introduction： 
This version will be updated recently.Please pay attension to this work.
The features are summarized blow:
+ Use VGG16 as base Network.
+ Dataset [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), you can split **trainset/testset** by yourself.
  Or you can download directly from [BaiduYun Link](https://pan.baidu.com/s/1JQxa3DYDrM329skC73kbzQ).
![Display](https://www.researchgate.net/profile/Xiangteng_He/publication/320032994/figure/fig1/AS:542681248288768@1506396700557/Examples-of-CUB-200-2011-dataset-1-First-row-shows-large-variance-in-the-same.png)
  
+ Part FCs is replaced by Global Average Pooling to reduce parameters.
+ Every epoch, ten best pathes is visualized in **vis_result** directory, you can put images you want to visualize in **vis_img** named number.jpg, e.g, 0.jpg.


### 2.Algorithms Introduction:
![Display](https://github.com/songdejia/DFL-CNN/blob/master/screenshot/introduction2.png)

![Display](https://github.com/songdejia/DFL-CNN/blob/master/screenshot/introduction1.jpg)


### 3.vis ten box for discriminative patch 
![Display](https://github.com/songdejia/DFL-CNN/blob/master/screenshot/vis_1.jpg)

![Display](https://github.com/songdejia/DFL-CNN/blob/master/screenshot/vis_2.jpg)

![Display](https://github.com/songdejia/DFL-CNN/blob/master/screenshot/vis_3.jpg)

![Display](https://github.com/songdejia/DFL-CNN/blob/master/screenshot/vis_4.jpg)


### 4.Usage:
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
+ Modify your run.sh 
```
sh run.sh
```


### 5.Note:
1. Visualization of ten best boxes is saved in **vis_result**. 
2. Weight(checkpoint.pth.tar, model_best.pth.tar) is in **weight**.
3. Loss info is saved in **log**.

