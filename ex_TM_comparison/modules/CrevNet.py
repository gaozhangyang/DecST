import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import functools
from torch.nn import init
import random
from itertools import accumulate


def split(x):
    n = int(x.size()[1]/2)
    x1 = x[:, :n, :, :].contiguous()
    x2 = x[:, n:, :, :].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)


class injective_pad(nn.Module):
    def __init__(self, pad_size):
        super(injective_pad, self).__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        l = len(x[1])
        return x[:, :l-self.pad_size, :, :]


class psi(nn.Module):
    def __init__(self, block_size):
        super(psi, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def inverse(self, inpt):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, d, h, w = inpt.shape
        return inpt.view(bs, bl, bl, int(d // bl_sq), h, w).permute(0, 3, 4, 1, 5, 2).reshape(bs, -1, h * bl, w * bl)

    def forward(self, inpt):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, d, new_h, new_w = inpt.shape[0], inpt.shape[1], int(inpt.shape[2] // bl), int(inpt.shape[3] // bl)
        return inpt.view(bs, d, new_h, bl, new_w, bl).permute(0, 3, 5, 1, 2, 4).reshape(bs, d * bl_sq, new_h, new_w)


class ListModule(object):
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


def get_all_params(var, all_params):
    if isinstance(var, Parameter):
        all_params[id(var)] = var.nelement()
    elif hasattr(var, "creator") and var.creator is not None:
        if var.creator.previous_functions is not None:
            for j in var.creator.previous_functions:
                get_all_params(j[0], all_params)
    elif hasattr(var, "previous_functions"):
        for j in var.previous_functions:
            get_all_params(j[0], all_params)


def transform_kitti(x):
    a = (x[:, 0] - 0.4182) #/ 0.28472
    b = (x[:, 1] - 0.43786) #/ 0.29514
    c = (x[:, 2] - 0.43288) #/ 0.30109
    return torch.cat((a[:,None,:,:],b[:,None,:,:],c[:,None,:,:]),1)

def transform_back_kitti(x):
    a = (x[:, 0] + 0.4182)
    b = (x[:, 1] + 0.43786)
    c = (x[:, 2] + 0.43288)
    return torch.cat((a[:,None,:,:],b[:,None,:,:],c[:,None,:,:]),1)


class irevnet_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, first=False, dropout_rate=0.,
                 affineBN=True, mult=4):
        """ buid invertible bottleneck block """
        super(irevnet_block, self).__init__()
        self.first = first
        self.stride = stride
        self.psi = psi(stride)
        layers = []
        if not first:
            layers.append(nn.GroupNorm(1,in_ch//2, affine=affineBN))
            layers.append(nn.ReLU(inplace=True))
        if int(out_ch//mult)==0:
            ch = 1
        else:
            ch =int(out_ch//mult)
        layers.append(nn.Conv2d(in_ch//2, ch, kernel_size=3,
                      stride=self.stride, padding=1, bias=False))
        layers.append(nn.GroupNorm(1,ch, affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(ch, ch,
                      kernel_size=3, padding=1, bias=False))
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.GroupNorm(1,ch, affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(ch, out_ch, kernel_size=3,
                      padding=1, bias=False))
        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        """ bijective or injective block forward """
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.bottleneck_block(x2)
        if self.stride == 2:
            x1 = self.psi.forward(x1)
            x2 = self.psi.forward(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        """ bijective or injecitve block inverse """
        x2, y1 = x[0], x[1]
        if self.stride == 2:
            x2 = self.psi.inverse(x2)
        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1
        if self.stride == 2:
            x1 = self.psi.inverse(x1)
        x = (x1, x2)
        return x


class STConvLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, memo_size):
        super(STConvLSTMCell,self).__init__()
        self.KERNEL_SIZE = 3
        self.PADDING = self.KERNEL_SIZE // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memo_size = memo_size
        self.in_gate = nn.Conv2d(input_size + hidden_size + hidden_size, hidden_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.remember_gate = nn.Conv2d(input_size + hidden_size+ hidden_size, hidden_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.cell_gate = nn.Conv2d(input_size + hidden_size+ hidden_size, hidden_size, self.KERNEL_SIZE, padding=self.PADDING)

        self.in_gate1 = nn.Conv2d(input_size + memo_size+ hidden_size, memo_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.remember_gate1 = nn.Conv2d(input_size + memo_size+ hidden_size, memo_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.cell_gate1 = nn.Conv2d(input_size + memo_size+ hidden_size, memo_size, self.KERNEL_SIZE, padding=self.PADDING)

        self.w1 = nn.Conv2d(hidden_size + memo_size, hidden_size, 1)
        self.out_gate = nn.Conv2d(input_size + hidden_size +hidden_size+memo_size, hidden_size, self.KERNEL_SIZE, padding=self.PADDING)


    def forward(self, input, prev_state):
        input_,prev_memo = input
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)).cuda(),
                Variable(torch.zeros(state_size)).cuda()
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden,prev_cell), 1)

        in_gate = F.sigmoid(self.in_gate(stacked_inputs))
        remember_gate =F.sigmoid(self.remember_gate(stacked_inputs))
        cell_gate = F.tanh(self.cell_gate(stacked_inputs))

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)

        stacked_inputs1 = torch.cat((input_, prev_memo,cell), 1)

        in_gate1 = F.sigmoid(self.in_gate1(stacked_inputs1))
        remember_gate1 = F.sigmoid(self.remember_gate1(stacked_inputs1))
        cell_gate1 = F.tanh(self.cell_gate1(stacked_inputs1))



        memo = (remember_gate1 * prev_memo) + (in_gate1 * cell_gate1)

        out_gate = F.sigmoid(self.out_gate(torch.cat((input_, prev_hidden,cell,memo), 1)))
        hidden = out_gate * F.tanh(self.w1(torch.cat((cell,memo),1)))
        #print(hidden.size())
        return (hidden, cell),memo




class zig_rev_predictor(nn.Module):
    def __init__(self, input_size, hidden_size,output_size,n_layers,batch_size, type = 'lstm', w = 8, h = 8):
        super(zig_rev_predictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.type = type
        self.w = w
        self.h = h

        self.convlstm = nn.ModuleList(
                [STConvLSTMCell(input_size, hidden_size,hidden_size) if i == 0 else STConvLSTMCell(hidden_size,hidden_size, hidden_size) for i in
                 range(self.n_layers)])

        self.att = nn.ModuleList([nn.Sequential(nn.Conv2d(self.hidden_size, self.hidden_size, 1, 1, 0),
                                                nn.GroupNorm(1,self.hidden_size,affine=True),
                                                nn.ReLU(),
                                                nn.Conv2d(self.hidden_size, self.hidden_size, 3, 1, 1),
                                                nn.GroupNorm(1, self.hidden_size, affine=True),
                                                nn.ReLU(),
                                                nn.Conv2d(self.hidden_size, self.hidden_size, 1, 1, 0),
                                                nn.Sigmoid()
                                                ) for i in range(self.n_layers)])

        self.hidden = self.init_hidden()
        self.prev_hidden = self.hidden

    def init_hidden(self,cc = 0):
        hidden = []

        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size,self.w,self.h).cuda(cc)),
                           Variable(torch.zeros(self.batch_size, self.hidden_size,self.w,self.h).cuda(cc))))
        return hidden

    def forward(self, input):
        input_, memo = input
        x1, x2 = input_
        mask = []
        for i in range(self.n_layers):
            out = self.convlstm[i]((x1,memo), self.hidden[i])
            self.hidden[i] = out[0]
            memo = out[1]
            g = self.att[i](self.hidden[i][0])
            mask.append(g)
            x2 = (1 - g) * x2 + g * self.hidden[i][0]
            x1, x2 = x2, x1

        return (x1,x2),memo,mask




class autoencoder(nn.Module):
    def __init__(self, nBlocks, nStrides, nChannels=None, init_ds=2,
                 dropout_rate=0., affineBN=True, in_shape=None, mult=4,it=False):
        super(autoencoder, self).__init__()
        self.ds = in_shape[2]//2**(nStrides.count(2)+init_ds//2)
        self.init_ds = init_ds
        if init_ds == 1:
            self.in_ch = in_shape[0]
        else:
            self.in_ch = in_shape[0] * 2**self.init_ds
        self.nBlocks = nBlocks
        self.first = True
        self.it = it
        # print('')
        # print(' == Building iRevNet %d == ' % (sum(nBlocks) * 3))
        if not nChannels:
            nChannels = [self.in_ch//2, self.in_ch//2 * 4,
                         self.in_ch//2 * 4**2, self.in_ch//2 * 4**3]

        self.init_psi = psi(self.init_ds)
        self.stack = self.irevnet_stack(irevnet_block, nChannels, nBlocks,
                                        nStrides, dropout_rate=dropout_rate,
                                        affineBN=affineBN, in_ch=self.in_ch,
                                        mult=mult)

    def irevnet_stack(self, _block, nChannels, nBlocks, nStrides, dropout_rate,
                      affineBN, in_ch, mult):
        """ Create stack of irevnet blocks """
        block_list = nn.ModuleList()
        strides = []
        channels = []
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            strides = strides + ([stride] + [1]*(depth-1))
            channels = channels + ([channel]*depth)
        for channel, stride in zip(channels, strides):
            block_list.append(_block(in_ch, channel, stride,
                                     first=self.first,
                                     dropout_rate=dropout_rate,
                                     affineBN=affineBN, mult=mult))
            in_ch = 2 * channel
            self.first = False
        return block_list

    def forward(self, input, is_predict = True):

        if is_predict:
            n = self.in_ch // 2
            if self.it:
                input = merge(input[0], input[1])
            if self.init_ds != 0:
                x = self.init_psi.forward(input)
            else:
                x = input
            out = (x[:, :n, :, :], x[:, n:, :, :])
            for block in self.stack:
                out = block.forward(out)
            x = out
        else:
            out = input
            for i in range(len(self.stack)):
                out = self.stack[-1 - i].inverse(out)
            out = merge(out[0], out[1])
            if self.init_ds != 0:
                x = self.init_psi.inverse(out)
            else:
                x = out
            if self.it:
                n = self.in_ch // 2
                x = (x[:, :n, :, :], x[:, n:, :, :])
        return x

class Dict(dict):
    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value

import sys
sys.path.append('/usr/data/gzy/Weather_Forecast')
import os
from tqdm import tqdm
import numpy as np

class CrevNet:
    def __init__(self,args):
        super(CrevNet,self).__init__()
        device = 'cuda:0'
        self.device = device
        self.opt=Dict({ 'lr':5e-4,
                        'rnn_size':384,
                        'g_dim':384,
                        'predictor_rnn_layers':8,
                        'batch_size':64,
                        'channels':3,
                        'image_width':128,
                        'image_height':128,
                        'n_past':4,
                        'n_future':4,
                        'beta1':0.9})

        self.encoder = autoencoder(nBlocks=[2,2,2,2], nStrides=[1, 2, 2, 2],
                            nChannels=None, init_ds=2,
                            dropout_rate=0., affineBN=True, in_shape=[self.opt.channels, self.opt.image_width, self.opt.image_height],
                            mult=4).to(device)

        self.frame_predictor = zig_rev_predictor(self.opt.rnn_size,  
                                                self.opt.rnn_size, 
                                                self.opt.g_dim, 
                                                self.opt.predictor_rnn_layers,
                                                self.opt.batch_size,
                                                'lstm',
                                                int(self.opt.image_width/16),
                                                int(self.opt.image_height/16)).to(device)

        self.optimizer = torch.optim.Adam([
                                        {'params': self.frame_predictor.parameters()},
                                        {'params': self.encoder.parameters()}
                                    ], lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

        self.criterion = nn.MSELoss()

        self.path = args.res_dir+'/{}'.format(args.ex_name)
        self.folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def train(self,train_loader, epoch):
        self.encoder.train()
        self.frame_predictor.train()
        train_pbar = tqdm(train_loader)
        for input_tensor, target_tensor,_ in train_pbar:
            if input_tensor.shape[0]<64:
                break
            self.optimizer.zero_grad()
            input_tensor = input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)
            input_tensor = torch.cat([input_tensor,target_tensor],dim=1)

            self.frame_predictor.hidden = self.frame_predictor.init_hidden()
            mse = 0
            memo = Variable(torch.zeros(self.opt.batch_size, self.opt.rnn_size, int(self.opt.image_width/16), int(self.opt.image_height/16)).cuda())

            for i in range(1, self.opt.n_past + self.opt.n_future):
                h = self.encoder(input_tensor[:,i - 1,::], True)
                h_pred,memo,_ = self.frame_predictor((h,memo))
                x_pred = self.encoder(h_pred, False)
                mse +=  self.criterion(x_pred, input_tensor[:,i,::])

            loss = mse
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def evaluate(self,test_loader):
        self.encoder.eval()
        self.frame_predictor.eval()
        preds = []
        trues = []
        test_pbar = tqdm(test_loader)
        with torch.no_grad():
            for input_tensor,target_tensor,_ in test_pbar:
                if input_tensor.shape[0]<64:
                    break
                #input_batch = torch.Size([8, 20, 1, 64, 64])
                input_tensor = input_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)
                input_tensor = torch.cat([input_tensor,target_tensor],dim=1)

                self.frame_predictor.hidden = self.frame_predictor.init_hidden()
                mse = 0
                memo = Variable(torch.zeros(self.opt.batch_size, self.opt.rnn_size, int(self.opt.image_width/16), int(self.opt.image_height/16)).cuda())
                pred_batch=[]
                for i in range(1, self.opt.n_past + self.opt.n_future):
                    h = self.encoder(input_tensor[:,i - 1,::], True)
                    h_pred,memo,_ = self.frame_predictor((h,memo))
                    x_pred = self.encoder(h_pred, False)
                    pred_batch.append(x_pred)
                
                pred_y = torch.stack(pred_batch,dim=1)[:,-self.opt.n_future:,:]
                pred_y = pred_y.cpu().numpy()
                target = target_tensor.cpu().numpy()
                preds.append(pred_y)
                trues.append(target)
        
        preds = np.concatenate(preds,axis=0)
        trues = np.concatenate(trues,axis=0)
        np.save(self.folder_path+'true.npy', trues)
        np.save(self.folder_path+'pred.npy', preds)

        import sys; sys.path.append('/usr/data/gzy/Weather_Forecast')
        from API.metrics import metric
        mae, mse, rmse, mape, mspe,ssim,psnr = metric(preds, trues,test_loader.dataset.mean,test_loader.dataset.std,return_ssim_psnr=True)
        return mse, mae, ssim


    def evaluate(self,test_loader):
        self.encoder.eval()
        self.frame_predictor.eval()
        preds = []
        trues = []
        test_pbar = tqdm(test_loader)
        number = 0
        with torch.no_grad():
            for input_tensor,target_tensor,_ in test_pbar:
                number+=input_tensor.shape[0]
                if input_tensor.shape[0]<64:
                    break
                #input_batch = torch.Size([8, 20, 1, 64, 64])
                input_tensor = input_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)
                input_tensor = torch.cat([input_tensor,target_tensor],dim=1)

                self.frame_predictor.hidden = self.frame_predictor.init_hidden()
                mse = 0
                memo = Variable(torch.zeros(self.opt.batch_size, self.opt.rnn_size, int(self.opt.image_width/16), int(self.opt.image_height/16)).cuda())
                pred_batch=[]
                for i in range(1, self.opt.n_past + self.opt.n_future):
                    h = self.encoder(input_tensor[:,i - 1,::], True)
                    h_pred,memo,_ = self.frame_predictor((h,memo))
                    x_pred = self.encoder(h_pred, False)
                    pred_batch.append(x_pred)
                
                pred_y = torch.stack(pred_batch,dim=1)[:,-self.opt.n_future:,:]
                pred_y = pred_y.cpu().numpy()
                target = target_tensor.cpu().numpy()
                preds.append(pred_y)
                trues.append(target)
                if number>=1000:
                    break
        
        preds = np.concatenate(preds,axis=0)
        trues = np.concatenate(trues,axis=0)
        np.save(self.folder_path+'true.npy', trues)
        np.save(self.folder_path+'pred.npy', preds)

        import sys; sys.path.append('/usr/data/gzy/Weather_Forecast')
        from API.metrics import metric
        mae, mse, rmse, mape, mspe,ssim,psnr = metric(preds, trues,test_loader.dataset.mean,test_loader.dataset.std,return_ssim_psnr=True)
        return mse, mae, ssim
    
    def validate(self,test_loader):
        self.encoder.eval()
        self.frame_predictor.eval()
        preds = []
        trues = []
        test_pbar = tqdm(test_loader)
        number = 0
        with torch.no_grad():
            for input_tensor,target_tensor,_ in test_pbar:
                number+=input_tensor.shape[0]
                if input_tensor.shape[0]<64:
                    break
                #input_batch = torch.Size([8, 20, 1, 64, 64])
                input_tensor = input_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)
                input_tensor = torch.cat([input_tensor,target_tensor],dim=1)

                self.frame_predictor.hidden = self.frame_predictor.init_hidden()
                mse = 0
                memo = Variable(torch.zeros(self.opt.batch_size, self.opt.rnn_size, int(self.opt.image_width/16), int(self.opt.image_height/16)).cuda())
                pred_batch=[]
                for i in range(1, self.opt.n_past + self.opt.n_future):
                    h = self.encoder(input_tensor[:,i - 1,::], True)
                    h_pred,memo,_ = self.frame_predictor((h,memo))
                    x_pred = self.encoder(h_pred, False)
                    pred_batch.append(x_pred)
                
                pred_y = torch.stack(pred_batch,dim=1)[:,-self.opt.n_future:,:]
                pred_y = pred_y.cpu().numpy()
                target = target_tensor.cpu().numpy()
                preds.append(pred_y)
                trues.append(target)
                if number>=1000:
                    break
        
        preds = np.concatenate(preds,axis=0)
        trues = np.concatenate(trues,axis=0)

        import sys; sys.path.append('/usr/data/gzy/Weather_Forecast')
        from API.metrics import metric
        mae, mse, rmse, mape, mspe = metric(preds, trues,test_loader.dataset.mean,test_loader.dataset.std)
        return mse, mae