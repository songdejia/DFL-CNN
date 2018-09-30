import torch
import datetime
import os
    
def get_root_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
   
def get_today():
    return str(datetime.date.today())
# gpu
def get_device_ids(num_gpu):
    assert isinstance(num_gpu, int),'num_gpu is not int'
    device_ids = []
    for i in range(num_gpu):
        device_ids.append(i)
    return device_ids 

# compute accurate
def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True) 
    pred = pred.t()            
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim = True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



#data load
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    # num of every class 
    for item in images:
        count[item[1]] += 1

    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])

    # weight for each image
    weight = [0]*len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight

class option_for_dataset_transform(object):
    def __init__(self, scale_width_keep_ar =None, random_crop =None, totensor=True, normalize=True):
        self.scale_width_keep_ar = scale_width_keep_ar
        self.random_crop = random_crop
        self.totensor = totensor
        self.normalize = normalize  

# network
def diagnose_network(net, name='network'):
    mean, count = 0.0, 0
    params = list(net.parameters())
    for layer, param in enumerate(net.parameters()):
        if param.grad is not None and layer < 10:
            print('layer: {:} '.format(layer))
            print('mean',torch.mean(torch.abs(param.data)))
            print('grad',torch.mean(torch.abs(param.grad.data)))

# train
def adjust_learning_rate(args, optimizer, epoch, gamma=0.1):
    """Sets the learning rate to the initial LR decayed 0.9 every 50 epochs"""
    lr = args.lr * (0.9 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr