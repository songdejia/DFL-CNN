import torch
import time
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
    for i, (data, target, paths) in enumerate(val_loader):
        if args.gpu is not None:
            data = data.cuda()
            target = target.cuda()

        # compute output
        output1, output2, output3, _ = model(data)
        loss1 = criterion(output1, target)
        loss2 = criterion(output2, target)
        loss3 = criterion(output3, target)
        loss = loss1 + loss2 + 0.1 * loss3
        output = output1 + output2 + 0.1 * output3

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    log.save_test_info(epoch, top1, top5)
    return top1.avg





