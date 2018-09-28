import torch
import time
import sys
from utils.util import *
from utils.save import *

def validate(args, val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    log = Log()
    model.eval()   
    end = time.time()

    # we may have ten d in data
    for i, (data, target, paths) in enumerate(val_loader):
        if args.gpu is not None:
            data = data.cuda()
            target = target.cuda()

        # compute output
        for idx, d in enumerate(data[0]):      # data [batchsize, 10_crop, 3, 448, 448]
            d = d.unsqueeze(0) # d [1, 3, 448, 448]
            output1, output2, output3, _ = model(d)
            output = output1 + output2 + 0.1 * output3

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            top1.update(prec1[0], 1)
            top5.update(prec5[0], 1)
            print('DFL-CNN <==> Test <==> Img:{} No:{} Top1 {:.3f} Top5 {:.3f}'.format(i, idx, prec1.cpu().numpy()[0], prec5.cpu().numpy()[0]))

    print('DFL-CNN <==> Test Total <==> Top1 {:.3f}% Top5 {:.3f}%'.format(top1.avg, top5.avg))
    log.save_test_info(epoch, top1, top5)
    return top1.avg




