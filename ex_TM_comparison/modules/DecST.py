import sys
sys.path.append('/usr/data/gzy/Weather_Forecast')
import torch.nn as nn
import torch
import numpy as np

class UnConv(nn.Module):
    def __init__(self,kernel_size,stride,padding):
        super(UnConv, self).__init__()
        self.k = kernel_size
        self.s = stride
        self.p = padding
    
    def forward(self,x):
        device = x.device
        B,C,H,W = x.shape
        W_out = 2*W
        k,s,p = self.k, self.s, self.k
        i,j=torch.meshgrid(torch.arange(H),torch.arange(W))
        indices = W_out*(k//2-p+s*i)+k//2-p+s*j
        Invert=torch.zeros(4*H*W,H*W)
        Invert[indices.view(-1),torch.arange(H*W)]=1
        Invert = Invert.to(device)
        y=torch.matmul(x.view(B,C,H*W),Invert.T).view(B,C,2*H,2*W)
        return y

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,transpose=False,speed_up=False,act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm=act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            if speed_up:
                self.conv = nn.Sequential(UnConv(kernel_size,stride,padding),
                                        nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=kernel_size//2))
            else:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=stride//2)
        self.norm = nn.GroupNorm(2,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,groups,act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm=act_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=1)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y

class ConvSC(nn.Module):
    def __init__(self, C_in,C_out, stride,  transpose=False):
        super(ConvSC,self).__init__()
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride, padding=1, transpose=transpose, act_norm=True)
    
    def forward(self,x):
        y = self.conv(x)
        return y


#------------------------ Inception layers
class gInception_ST(nn.Module):
    def __init__(self, T_in,C_in,T_out,C_out, dec=False):        
        super(gInception_ST, self).__init__()
        ksize=[3,5,7,11]
        stride=[1,1,1,1]
        self.T_in, self.C_in, self.T_out, self.C_out = T_in,C_in,T_out,C_out
        in_channels, out_channels = T_in*C_in, T_out*C_out
        # groups = C_in if (out_channels // 2) % C_in==0 else C_in//2
        groups = 8
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.conv_3x3 = GroupConv2d(out_channels, out_channels, kernel_size=ksize[0], stride=stride[0],padding=ksize[0]//2, groups=groups, act_norm=True)
        self.conv_5x5 = GroupConv2d(out_channels, out_channels, kernel_size=ksize[1], stride=stride[1], padding=ksize[1]//2,groups=groups, act_norm=True)
        self.conv_7x7 = GroupConv2d(out_channels, out_channels, kernel_size=ksize[2], stride=stride[2], padding=ksize[2]//2,groups=groups, act_norm=True)   
        self.conv_11x11 = GroupConv2d(out_channels, out_channels, kernel_size=ksize[3], stride=stride[3], padding=ksize[3]//2,groups=groups, act_norm=True)  

    def forward(self, x):
        x = self.conv1(x)
        return self.conv_3x3(x) + self.conv_5x5(x) + self.conv_7x7(x)+self.conv_11x11(x)



#--------------------------- encoder
class EncoderShallow(nn.Module):
    def __init__(self,C_in,C_hid):
        super(EncoderShallow,self).__init__()
        self.enc1 = ConvSC(C_in, C_hid, stride=1)      
        self.enc2 = ConvSC(C_hid, 2*C_hid, stride=2)   
        self.enc3 = ConvSC(2*C_hid, 2*C_hid, stride=1) 
    
    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc1(x)    #     (4,16,32,32)
        enc2 = self.enc2(enc1) #     (4,32,16,16)
        latent = self.enc3(enc2) #     (4,32,16,16)
        return latent,[enc2,enc1]


class EncoderDeep(nn.Module):
    def __init__(self,C_in,C_hid):
        super(EncoderDeep,self).__init__()
        self.enc1 = ConvSC(C_in, C_hid, stride=1)     
        self.enc2 = ConvSC(C_hid, 2*C_hid, stride=1)  
        self.enc3 = ConvSC(2*C_hid, 4*C_hid, stride=2) 
        self.enc4 = ConvSC(4*C_hid, 4*C_hid, stride=1) 
        self.enc5 = ConvSC(4*C_hid, 8*C_hid, stride=2) 
        self.enc6 = ConvSC(8*C_hid, 8*C_hid, stride=1) 
    
    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc1(x)    #     16, 128, 128
        enc2 = self.enc2(enc1) #     32, 128, 128
        enc3 = self.enc3(enc2) #     64,  64,  64
        enc4 = self.enc4(enc3) #     64,  64,  64
        enc5 = self.enc5(enc4) #     128, 32,  32
        latent = self.enc6(enc5)  #     128, 32,  32
        return latent,[enc5,enc4,enc3,enc2,enc1]


#--------------------------- time module
class Mid_Xnet(nn.Module):
    def __init__(self, T_in,C_in,T_hid,C_hid):
        super(Mid_Xnet, self).__init__()
        self.T_in ,self.C_in ,self.T_hid ,self.C_hid = T_in,C_in,T_hid,C_hid
        self.enc1 = gInception_ST(T_in,C_in,T_hid,C_hid)
        self.enc2 = gInception_ST(T_hid,C_hid,T_hid,C_hid)
        self.enc3 = gInception_ST(T_hid,C_hid,T_hid,C_hid)
        self.dec3 = gInception_ST(2*T_hid,C_hid,T_hid,C_hid, dec=True)
        self.dec2 = gInception_ST(2*T_hid,C_hid,T_hid,C_hid, dec=True)
        self.dec1 = gInception_ST(2*T_hid,C_hid,T_in,C_in, dec=True)

    def forward(self, x):
        enc1 = self.enc1(x) #[64, 160, 16, 16]
        enc2 = self.enc2(enc1) #[64, 128, 16, 16]
        enc3 = self.enc3(enc2) #[64, 128, 16, 16]

        dec3 = self.dec3(torch.cat([enc3, enc2], dim=1)) #[64, 128, 16, 16]
        dec2 = self.dec2(torch.cat([dec3, enc1+enc2+enc3], dim=1)) #[64, 128, 16, 16]
        dec1 = self.dec1(torch.cat([dec2, enc1+enc2], dim=1)) #[64, 128, 16, 16]
        return dec1


#--------------------------- decoder
class DecoderShallow(nn.Module):
    def __init__(self,C_in,C_hid,unet=True):
        super(DecoderShallow,self).__init__()
        self.unet = unet
        self.dec3 = ConvSC(2*C_hid, 2*C_hid,stride=1)
        self.dec2 = ConvSC(2*2*C_hid, C_hid, stride=2, transpose=True)
        self.dec1 = ConvSC(2*C_hid, C_hid, stride=1)
        self.readout_f = nn.Conv2d(C_hid, C_in, 1)
    
    def forward(self,hid,skip=None):
        dec3 = self.dec3(hid)  # (4,32,16,16)
        dec2 = self.dec2(torch.cat([dec3, skip[0]], dim=1)) # (4,16,32,32)
        dec1 = self.dec1(torch.cat([dec2, skip[1]], dim=1)) # (4,16,32,32)
        
        Y_f = self.readout_f(dec1)
        return Y_f


class DecoderDeep(nn.Module):
    def __init__(self,C_in,C_hid,unet=True):
        super(DecoderDeep,self).__init__()
        self.unet = unet
        self.dec6 = ConvSC(8*C_hid, 8*C_hid, stride=1)
        self.dec5 = ConvSC(2*8*C_hid, 4*C_hid, stride=2, transpose=True)
        self.dec4 = ConvSC(2*4*C_hid, 4*C_hid,stride=1)
        self.dec3 = ConvSC(2*4*C_hid, 2*C_hid, stride=2,  transpose=True)
        self.dec2 = ConvSC(2*2*C_hid, C_hid, stride=1)
        self.dec1 = ConvSC(2*C_hid, C_hid, stride=1)
        self.readout_f = nn.Conv2d(C_hid, C_in, 1)
    
    def forward(self,hid,skip=None):
        dec6 = self.dec6(hid)  # (128M + 128E), 32, 32 -> 128, 32, 32
        dec5 = self.dec5(torch.cat([dec6, skip[0]], dim=1)) # (128D + 128E), 32, 32 -> 64, 64, 64
        dec4 = self.dec4(torch.cat([dec5, skip[1]], dim=1)) # (64D + 64E), 64, 64   -> 64, 64, 64
        dec3 = self.dec3(torch.cat([dec4, skip[2]], dim=1)) # (64D + 64E), 64, 64   -> 32, 128, 128
        dec2 = self.dec2(torch.cat([dec3, skip[3]], dim=1)) # (32D + 32E), 128, 128 -> 16, 128, 128
        dec1 = self.dec1(torch.cat([dec2, skip[4]], dim=1)) # (16D + 16E), 128, 128 ->  4, 128, 128
        
        Y = self.readout_f(dec1)
        return Y


class ConvUnet(nn.Module):
    def __init__(self, dataname,shape_in,C_hid=16, T_hid=2):
        super(ConvUnet,self).__init__()
        self.dataname = dataname
        T, C, H, W = shape_in

        self.enc = EncoderDeep(C,C_hid)
        self.hid = Mid_Xnet(T,8*C_hid,T_hid,8*C_hid)
        self.dec = DecoderDeep(C,C_hid,unet=True)
    
    def forward(self,x_raw):
        B,T,C,H,W = x_raw.shape
        x = x_raw.view(B*T,C,H,W)
        embed,skip = self.enc(x)
        _,C_,H_,W_ = embed.shape
        self.embed = embed

        z = embed.view(B,T*C_,H_,W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T,C_,H_,W_)

        Y = self.dec(hid,skip)
        Y = Y.reshape(B,T,C,H,W)
        return Y


import os
from tqdm import tqdm

class DecST:
    def __init__(self, args):
        super(DecST, self).__init__()
        device = 'cuda:0'
        self.device = device
        self.model = ConvUnet('human', (4,3,128,128),C_hid=16, T_hid=4).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.01)
        self.criterion = nn.MSELoss()

        self.path = args.res_dir+'/{}'.format(args.ex_name)
        self.folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def train(self,train_loader, epoch): 
        self.model.train()

        train_pbar = tqdm(train_loader)
        for input_tensor, target_tensor,_ in train_pbar:
            self.optimizer.zero_grad()
            # input_tensor : torch.Size([batch_size, input_length, 1, 64, 64])
            input_tensor = input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)
            
            pred_y = self.model(input_tensor)
            loss = self.criterion(pred_y, target_tensor)

            loss.backward()
            self.optimizer.step()
        return loss.item()
    

    def evaluate(self,test_loader):
        self.model.eval()
        preds = []
        trues = []
        test_pbar = tqdm(test_loader)
        with torch.no_grad():
            for input_tensor,target_tensor,_ in test_pbar:
                #input_batch = torch.Size([8, 20, 1, 64, 64])
                input_tensor = input_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)

                pred_y = self.model(input_tensor)

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
    
    def validate(self,test_loader):
        self.model.eval()
        preds = []
        trues = []
        test_pbar = tqdm(test_loader)
        number=0
        with torch.no_grad():
            for input_tensor,target_tensor,_ in test_pbar:
                #input_batch = torch.Size([8, 20, 1, 64, 64])
                number+=input_tensor.shape[0]
                input_tensor = input_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)

                pred_y = self.model(input_tensor)

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

if __name__ == '__main__':
    device = torch.device('cuda:0')
    T, C, H, W = 10,1,64,64
    x = torch.rand(64, T, C, H, W).to(device)
    model = ConvUnet('mmist',(T,C,H,W)).to(device)
    Y = model(x)
    print()
