import numpy as np
import torch
import torchvision 
from torchvision import datasets, transforms, utils
import sys

if __name__ == '__main__':  
    BS = 1028
    transform_ = transforms.Compose([
        transforms.Resize((96,128)),
        transforms.ToTensor()
    ])
    path ='/data1/data_sdj/data/new_data_100400/train'
    dataset = datasets.ImageFolder(path, transform = transform_)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BS, drop_last = True,num_workers=4)
    m,num = 0, 0
    for i, (data, label)in enumerate(dataloader):
        m += data
        num += BS
        print(i)
        #print(np.mean(data, axis=(0,1,2)))
        #print(data.std(axis=(0,1,2)))
    mean = m/num
    a = mean[0]
"""
class container(object):
    def __init__(self):
        self.val = 0
        self.sum = 0
        self.num = 0
        self.avg = 0

    def update(self, val, num):
        self.val = val
        self.num += num
        self.sum += num*val
        self.avg = self.sum/self.num

"""
