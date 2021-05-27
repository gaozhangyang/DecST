
import sys; sys.path.append('..')
from API.tools import EarlyStopping
from API.exp_basic import Exp_Basic

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from ex_archi_search.model import ConvUnet,DecST

from API.dataloader import load_data
import json

import os
import time
import logging
from tqdm import tqdm
from API.metrics import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau
import nni

import warnings
warnings.filterwarnings('ignore')

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

        self._select_optimizer()
        if self.args.epoch_s>0:
            self._load(self.args.epoch_s-1)
    
    def _build_model(self):
        if self.args.method=='DecST':
            model = DecST(tuple(self.args.in_shape))
        else:
            model = ConvUnet(self.args.method,tuple(self.args.in_shape),self.args.UNet,self.args.XNet,self.args.gConv,self.args.Incep)
        return model

    def _get_data(self):
        config = self.args.__dict__

        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(config['dataname'],config['batch_size'], config['val_batch_size'], config['data_root'])
        if self.vali_loader is None:
            self.vali_loader = self.test_loader

    def _select_optimizer(self):
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = ReduceLROnPlateau(self.model_optim, mode='min', patience=3,factor=0.5,verbose=True)
        return self.model_optim
    
    def _adjust_learning_rate(self,optimizer,epoch,args):
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch-1) // 2))}

        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion
    
    def _save(self,epoch):
        torch.save(self.model.state_dict(), os.path.join(self.checkpoints_path, str(epoch) + '.pth'))
        state=self.scheduler.state_dict()
        with open(os.path.join(self.checkpoints_path, str(epoch) + '.json'), 'w') as file_obj:
            json.dump(state, file_obj)
    
    def _load(self,epoch):
        self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_path, str(epoch) + '.pth')))
        state = json.load(open(self.checkpoints_path+'/'+str(epoch) + '.json','r'))
        self.scheduler.load_state_dict(state)

    def vali(self, vali_loader, criterion, name,epoch):
        self.model.eval()
        preds=[]
        trues=[]
        total_loss = []
        vali_pbar = tqdm(vali_loader)
        for batch_x,batch_y in vali_pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y

            pred_y,loss2 = self.model(batch_x)
            true = batch_y.detach().cpu()
            pred_y = pred_y.detach().cpu()
            loss = criterion(pred_y, true)
            vali_pbar.set_description('vali loss: {:.4f}'.format(loss.item()))
            total_loss.append(loss)

            preds.append(pred_y.numpy())
            trues.append(true.numpy())

        total_loss = np.average(total_loss)

        preds = np.concatenate(preds,axis=0)
        trues = np.concatenate(trues,axis=0)
        mae, mse, rmse, mape, mspe = metric(preds, trues,vali_loader.dataset.mean,vali_loader.dataset.std)
        print('{}\tmse:{}, mae:{}, rmse:{}, mape:{} ,mspe:{}'.format(name,mse, mae, rmse, mape, mspe ))
        logging.info('{}\tmse:{}, mae:{}, rmse:{}, mape:{} ,mspe:{}'.format(name,mse, mae, rmse, mape, mspe ))
        self.model.train()

        if name == 'vali':
            nni.report_intermediate_result(mse)

        return total_loss
    

    def train(self, args):
        config = args.__dict__
        time_now = time.time()
        
        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        for epoch in range(config['epoch_s'], config['epoch_e']):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            train_pbar = tqdm(self.train_loader)
            i=0
            for batch_x,batch_y in train_pbar:
                iter_count += 1
                
                model_optim.zero_grad()
                batch_x = batch_x.to(self.device) # [32,12,3,32,64]
                batch_y = batch_y.to(self.device) # [32,12,3,32,64]

                pred_y, loss2 = self.model(batch_x)
                loss = criterion(pred_y, batch_y) + loss2
                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
                
                loss.backward()
                model_optim.step()
                i+=1

            train_loss = np.average(train_loss)
            if epoch % args.log_step == 0:
                self._save(epoch)
                vali_loss = self.vali(self.vali_loader, criterion,'vali',epoch)
                test_loss = self.vali(self.test_loader, criterion,'test',epoch)
                self.scheduler.step(test_loss)
                # nni.report_intermediate_result(test_loss)


                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}\n".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                logging.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}\n".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, self.path)

            if early_stopping.early_stop:
                print("Early stopping")
                logging.info("Early stopping")
                break
            
        best_model_path = self.path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self,args):
        self.model.eval()
        preds = []
        trues = []
        Y_f = []
        Y_b = []
        mask = []
        inputs = []
        
        for batch_x,batch_y in self.test_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y

            pred_y, loss2 = self.model(batch_x)#.squeeze()
            
            inputs.append(batch_x.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
            preds.append(pred_y.detach().cpu().numpy())
            Y_f.append(self.model.Y_f.detach().cpu().numpy())
            Y_b.append(self.model.Y_b.detach().cpu().numpy())
            mask.append(self.model.mask.detach().cpu().numpy())
            
        inputs = np.concatenate(inputs,axis=0)
        trues = np.concatenate(trues,axis=0)
        preds = np.concatenate(preds,axis=0)
        Y_f = np.concatenate(Y_f,axis=0)
        Y_b = np.concatenate(Y_b,axis=0)
        mask = np.concatenate(mask,axis=0)
        
        print('test shape:', preds.shape, trues.shape)
        logging.info('test shape:{}-{}'.format(preds.shape, trues.shape))

        # result save
        folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues,self.test_loader.dataset.mean,self.test_loader.dataset.std)
        print('mse:{}, mae:{}'.format(mse, mae))
        logging.info('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'inputs.npy', inputs)
        np.save(folder_path+'true.npy', trues)
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'Y_f.npy', Y_f)
        np.save(folder_path+'Y_b.npy', Y_b)
        np.save(folder_path+'mask.npy', mask)
        
        return mse