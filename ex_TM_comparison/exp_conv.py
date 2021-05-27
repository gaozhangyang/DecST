
import sys; sys.path.append('..')
from ex_TM_comparison.exp_basic import Exp_Basic

import numpy as np
import torch
import torch.nn as nn
from API.dataloader import load_data
import json

import os
import time
import logging
import nni
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Exp_Traffic(Exp_Basic):
    def __init__(self, args):
        super(Exp_Traffic, self).__init__(args)
        self.path = args.res_dir+'/{}'.format(args.ex_name)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.checkpoints_path = os.path.join(self.path, 'checkpoints')
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

        sv_param = os.path.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(args.__dict__, file_obj)
        
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                            filename=self.path+'/log.log',#'log/{}_{}_{}.log'.format(args.gcn_type,args.graph_type,args.order_list)
                            filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                            #a是追加模式，默认如果不写的话，就是追加模式
                            format='%(asctime)s - %(message)s'#日志格式
                            )
    
        self._get_data()
        self._build_model()
        # self.NParam = count_parameters(self.algotirhm.model)
        # print('{}'.format(self.NParam))
        # logging.info('{}'.format(self.NParam))

        if self.args.epoch_s>0:
            self._load(self.args.epoch_s-1)
    
    def _build_model(self):
        # ################################################TODO 写好不同模型的API
        if self.args.model=='PhyDNet':
            from ex_TM_comparison.modules.PhyDNet import PhyDNet
            self.algotirhm = PhyDNet(self.args)

        if self.args.model=='DecST':
            from ex_TM_comparison.modules.DecST import DecST
            self.algotirhm = DecST(self.args)
        
        if self.args.model=='PredRNN':
            from ex_TM_comparison.modules.predrnn import PredRNN
            self.algotirhm = PredRNN(self.args)
        
        if self.args.model=='ConvLSTM':
            from ex_TM_comparison.modules.convlstm import ConvLSTM
            self.algotirhm = ConvLSTM(self.args)

        if self.args.model=='E3DLSTM':
            from ex_TM_comparison.modules.e3d_lstm import E3DLSTM
            self.algotirhm = E3DLSTM(self.args)

        if self.args.model=='MIM':
            from ex_TM_comparison.modules.mim import MIM
            self.algotirhm = MIM(self.args)
        
        if self.args.model == 'CrevNet':
            from ex_TM_comparison.modules.CrevNet import CrevNet
            self.algotirhm = CrevNet(self.args)
        
        return self.algotirhm 

    def _get_data(self):
        config = self.args.__dict__

        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(config['dataname'],config['batch_size'], config['val_batch_size'], config['data_root'],require_back=True)
        if self.vali_loader is None:
            self.vali_loader = self.test_loader
    
    def _save(self,epoch):
        torch.save(self.algotirhm.model.state_dict(), os.path.join(self.checkpoints_path, str(epoch) + '.pth'))
        state=self.scheduler.state_dict()
        with open(os.path.join(self.checkpoints_path, str(epoch) + '.json'), 'w') as file_obj:
            json.dump(state, file_obj)
    
    def _load(self,epoch):
        self.algotirhm.model.load_state_dict(torch.load(os.path.join(self.checkpoints_path, str(epoch) + '.pth')))
        state = json.load(open(self.checkpoints_path+'/'+str(epoch) + '.json','r'))
        self.scheduler.load_state_dict(state)

    def train(self, args):
        config = args.__dict__
        sv_info = pd.DataFrame(columns=['epoch','train_time','eval_time','mse','mae'])
        for epoch in range(config['epoch_s'], config['epoch_e']):
            t0 = time.time()
            mse_loss = self.algotirhm.train(self.train_loader, epoch)
            t1 = time.time()

            mse, mae = self.algotirhm.validate(self.test_loader)
            t2 = time.time()

            sv_info = sv_info.append({'epoch':epoch,'train_time':t1-t0,'eval_time':t2-t1, 'mse':mse,'mae':mae}, ignore_index=True)
            nni.report_intermediate_result(mse)

        sv_info.to_csv(self.path+'/sv_info.csv',index=False)

        mse, mae, ssim = self.algotirhm.evaluate(self.test_loader)
        logging.info('mse:{}\tmae:{}\tssim:{}'.format(mse,mae,ssim))
        print('mse:{}\tmae:{}\tssim:{}'.format(mse,mae,ssim))
        nni.report_final_result(mse)
        return mse_loss
