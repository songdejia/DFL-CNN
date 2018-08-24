from model.DFL import DFL_VGG
from utils.util import *
from utils.transform import *
from train import *
from validate import *
from utils.init import *
import sys
import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.MyImageFolderWithPaths import *
from drawrect import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataroot', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--result', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-trainbatch', '--train_batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('-testbatch', '--test_batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--init_type',  default='xavier', type=str,
                    metavar='INIT',help='init net')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='momentum', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.000005, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=4, type=int,
                    help='GPU nums to use.')
parser.add_argument('--log_train_dir', default='log_train', type=str,
                    help='log for train')
parser.add_argument('--log_test_dir', default='log_test', type=str,
                    help='log for test')
parser.add_argument('--nclass', default=583, type=int,
                    help='num of classes')
parser.add_argument('--w', default=448, type=int,
                    help='transform, seen as align')
parser.add_argument('--h', default=448, type=int,
                    help='transform, seen as align')


best_prec1 = 0

def main():
     global args, best_prec1
     args = parser.parse_args()
     
     # DFL Model
     model = DFL_VGG(k = 10, nclass = 200)     
     
     if args.gpu is not None:
        ids = get_device_ids(args.gpu)
        model = nn.DataParallel(model, device_ids=ids)
        model = model.cuda()

     if args.init_type is not None:
          init_weights(model, init_type=args.init_type)
     print('Model is Created and Initialized')
     
     # Define loss function (criterion) and optimizer
     criterion = nn.CrossEntropyLoss()
     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)
     print('Criterion and Optimizer is Done')

     # Optionally resume from a checkpoint
     if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

     cudnn.benchmark = True

     # Data loading code
     dataroot = os.path.abspath(args.dataroot)
     traindir = os.path.join(dataroot, 'train')
     testdir = os.path.join(dataroot, 'test')

     # ImageFolder to process img
     transform = get_transform()
     train_dataset = ImageFolderWithPaths(traindir, transform = transform)
     test_dataset  = ImageFolderWithPaths(testdir, transform = transform)

     # A list for target to classname
     index2classlist = train_dataset.index2classlist()

     # data loader   
     train_loader = torch.utils.data.DataLoader(
          train_dataset, batch_size=args.train_batch_size, shuffle=True,
          num_workers=args.workers, pin_memory=True, drop_last = False)
     test_loader = torch.utils.data.DataLoader(
          test_dataset, batch_size=args.test_batch_size, shuffle=True,
          num_workers=args.workers, pin_memory=True, drop_last = False)
     print('data loader Done')
   

       
     for epoch in range(args.start_epoch, args.epochs):
        """
        for data, labels, paths in train_loader:
            check_classname2dirname(index2classlist, labels, paths)
        """
        adjust_learning_rate(args, optimizer, epoch, gamma = 0.1)
        
        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch)
        
        # evaluate on validation set
        prec1 = validate(args, test_loader, model, criterion, epoch)
        
        # do a test for visualization
        draw_patch(epoch, model, index2classlist, args.result)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

if __name__ == '__main__':
     main() 
