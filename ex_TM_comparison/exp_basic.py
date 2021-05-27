import os
import torch
import numpy as np

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        print(args)
        self.device = self._acquire_device()

    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:0')
            print('Use GPU:',device)
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass