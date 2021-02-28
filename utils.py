import argparse
import sys
import os
from collections import defaultdict

# import importlib.machinery
# anaconda3_path = os.path.abspath(sys.executable + "/../../")
# pynvml_path = anaconda3_path + '/lib/python3.7/site-packages/'
# sys.path.append(pynvml_path)
#
# loader = importlib.machinery.SourceFileLoader('my_pynvml', pynvml_path + 'pynvml.py')
# my_pynvml = loader.load_module()
# import pynvml in anaconda
# import my_pynvml

# def get_default_gpus(parallels):
#     my_pynvml.nvmlInit()
#     deviceCount = my_pynvml.nvmlDeviceGetCount()
#
#     gpu_usage = defaultdict(int)
#     for i in range(deviceCount):
#         handle = my_pynvml.nvmlDeviceGetHandleByIndex(i)
#         info = my_pynvml.nvmlDeviceGetMemoryInfo(handle)
#
#         gpu_usage[i] = info.used
#
#     sorted_gpu_usage = sorted(gpu_usage, key=gpu_usage.get)
#
#     return sorted_gpu_usage[:parallels]

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
