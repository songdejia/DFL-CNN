# DFL-
This is a pytorch re-implementation of [Learning a Discriminative Filter Bank Within a CNN for Fine-Grained Recognition](https://arxiv.org/pdf/1611.09932.pdf) 

The features are summarized blow:
1. Use VGG16 as base Network
2. Dataset CUB-200-2011
3. Part FCs is replaced by GAP to reduce parameters which you can modify by yourself
4. Every epoch, ten best pathes is visualized in result directory

Dataset:[CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

![Display](https://www.researchgate.net/profile/Xiangteng_He/publication/320032994/figure/fig1/AS:542681248288768@1506396700557/Examples-of-CUB-200-2011-dataset-1-First-row-shows-large-variance-in-the-same.png)

Usage:
1. wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
2. tar -zxvf CUB-200-2011.tgz
3. Then, split into two directorys by train_test_split.txt as 
  root/train/classx
  ...
  root/test/classx
4. Finally, 
 ln -s ./train path/to/code/dataset/train &&
 ln -s ./test  path/to/code/dataset/test

Run:
1. Modufy your run.sh 
2. sh run.sh
Visualization of ten best boxes is saved in result
weight(checkpoint.pth.tar, model_best.pth.tar) is in ./weight
loss info is saved in log

