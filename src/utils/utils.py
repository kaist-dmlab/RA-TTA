import logging
import os
import random
import shutil
import time
from enum import Enum

import numpy as np
import torch

logger = logging.getLogger('RATTA')

# Utils

import contextlib

@contextlib.contextmanager
def fork_rng_with_seed(seed):
    if seed is None:
        yield
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield


def set_seed(random_seed):
    logger.info('Set seed to %d.' % random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    # fix the seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU


def get_env_info():
    """
    Get environment information of pytorch and torchvision
    """

    import torch
    import torchvision

    version_info = ('\nVersion Information: '
            f'\n\tPyTorch: {torch.__version__}'
            f'\n\tTorchVision: {torchvision.__version__}')
    return version_info

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def gpu_list():
    # Returns the index of a currently selected device.
    return torch.cuda.current_device()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct


def count_acc(logits, labels):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == labels).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == labels).type(torch.FloatTensor).mean().item()


def get_result(logits, labels):
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).type(torch.cuda.FloatTensor)


def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else:
        import argparse
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_checkpoint(state, is_best, save_dir=None):
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, checkpoint_path)
    if is_best:
        best_checkpoint_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()


def count_acc_topk(x,y,k=5):
    _,maxk = torch.topk(x,k,dim=-1)
    total = y.size(0)
    test_labels = y.view(-1,1) 
    #top1=(test_labels == maxk[:,0:1]).sum().item()
    topk=(test_labels == maxk).sum().item()
    return float(topk/total)


def _plot_task(axs, images, n_samples, mode='gray'):
    for sample in range(n_samples):
        axs[sample].imshow(images[sample], cmap="gray")
        axs[sample].xaxis.set_ticks([])
        axs[sample].yaxis.set_ticks([])

def print_latex_table(lst):
    length = len(lst)
    output = ""
    for i in range(length):
        output += str(lst[i]) 
        if i+1 != length:
            output += ' & '
    return output 

def detach_and_clone(obj):
    if torch.is_tensor(obj):
        return obj.detach().clone()
    elif isinstance(obj, dict):
        return {k: detach_and_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_and_clone(v) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        raise TypeError("Invalid type for detach_and_clone")

def collate_list(vec):
    """
    If vec is a list of Tensors, it concatenates them all along the first dimension.

    If vec is a list of lists, it joins these lists together, but does not attempt to
    recursively collate. This allows each element of the list to be, e.g., its own dict.

    If vec is a list of dicts (with the same keys in each dict), it returns a single dict
    with the same keys. For each key, it recursively collates all entries in the list.
    """
    if not isinstance(vec, list):
        raise TypeError("collate_list must take in a list")
    elem = vec[0]
    if torch.is_tensor(elem):
        return torch.cat(vec)
    elif isinstance(elem, list):
        return [obj for sublist in vec for obj in sublist]
    elif isinstance(elem, dict):
        return {k: collate_list([d[k] for d in vec]) for k in elem}
    else:
        raise TypeError("Elements of the list to collate must be tensors or dicts.")
