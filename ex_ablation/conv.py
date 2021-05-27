import torch
from torch import nn
import time

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


class UnConv2d(nn.Module):
    def __init__(self,C_in,C_out,stride_out,kernel_size,stride,padding):
        super(UnConv2d, self).__init__()
        self.unconv = UnConv(kernel_size,stride,padding)
        self.conv = nn.Conv2d(C_in,C_out,stride_out)
    
    def forward(self,x):
        x = self.unconv(x)
        y = self.conv(x)
        return y


##--------------------------------------------- Basic convolution
class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False,act_norm=False):
        super(BasicConv1d, self).__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=stride//2)
        self.norm = nn.GroupNorm(2,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            self.activate(self.norm(y))
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
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,transpose=False,output_padding=(0,0,0),act_norm=False):
        super(BasicConv3d, self).__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose3d(in_channels,out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=output_padding)
        self.norm = nn.GroupNorm(2,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


##-------------------------------------------- DecConv3D
class Conv3D_(nn.Module):
    def __init__(self,in_shape,out_shape,reverse=False,act_norm=False):
        super(Conv3D_,self).__init__()
        T,C,H,W = in_shape
        T_,C_,H_,W_ = out_shape
        self.in_shape = in_shape
        self.out_shape = out_shape
        transposeT = True if T_ > T else False
        transposeS = True if W_ > W else False
        strideT = int(max(T/T_,T_/T))
        strideS = int(max(W/W_,W_/W))
        outpadS = 1 if transposeS else 0
        outpadT = 1 if transposeT else 0
        self.conv1 = BasicConv3d(C,C_,3,(strideT,strideS,strideS),1,transposeS,output_padding=(outpadT,outpadS,outpadS),act_norm=act_norm)
        
        self.t1 = 0
        self.count = 0
    
    def forward(self,x):
        B,T,C,H,W = x.shape
        x = x.permute(0,2,1,3,4).contiguous()

        t1 = time.time()
        y = self.conv1(x)
        t2 = time.time()
        y = y.permute(0,2,1,3,4).contiguous()

        self.t1+=t2-t1
        self.count+=1
        return y


