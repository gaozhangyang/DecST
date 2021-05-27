import sys

from numpy.core.fromnumeric import transpose; sys.path.append('/usr/data/gzy/Weather_Forecast')
from ex_sota.conv import BasicConv1d, BasicConv2d, BasicConv3d, GroupConv2d
import torch.nn as nn
import torch
import torch.nn.functional as F

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
        if unet:
            self.dec3 = ConvSC(2*C_hid, 2*C_hid,stride=1)
            self.dec2 = ConvSC(2*2*C_hid, C_hid, stride=2, transpose=True)
            self.dec1 = ConvSC(2*C_hid, C_hid, stride=1)
            self.readout_f = nn.Conv2d(C_hid, C_in, 1)
            self.readout_m = nn.Conv2d(C_hid, 1, 1)
        else:
            self.dec3 = ConvSC(2*C_hid, 2*C_hid,stride=1)
            self.dec2 = ConvSC(2*C_hid, C_hid, stride=2, transpose=True)
            self.dec1 = ConvSC(C_hid, C_hid, stride=1)
            self.readout_f = nn.Conv2d(C_hid, C_in, 1)
            self.readout_m = nn.Conv2d(C_hid, 1, 1)
    
    def forward(self,hid,skip=None):
        if self.unet:
            dec3 = self.dec3(hid)  # (4,32,16,16)
            dec2 = self.dec2(torch.cat([dec3, skip[0]], dim=1)) # (4,16,32,32)
            dec1 = self.dec1(torch.cat([dec2, skip[1]], dim=1)) # (4,16,32,32)
            
            Y_f = self.readout_f(dec1)
            mask = torch.sigmoid( self.readout_m(dec1) )
        else:
            dec3 = self.dec3(hid)  # (4,32,16,16)
            dec2 = self.dec2(dec3) # (4,16,32,32)
            dec1 = self.dec1(dec2) # (4,16,32,32)
            
            Y_f = self.readout_f(dec1)
            mask = torch.sigmoid( self.readout_m(dec1) )
        return Y_f,mask


class DecoderDeep(nn.Module):
    def __init__(self,C_in,C_hid,unet=True):
        super(DecoderDeep,self).__init__()
        self.unet = unet
        if unet:
            self.dec6 = ConvSC(8*C_hid, 8*C_hid, stride=1)
            self.dec5 = ConvSC(2*8*C_hid, 4*C_hid, stride=2, transpose=True)
            self.dec4 = ConvSC(2*4*C_hid, 4*C_hid,stride=1)
            self.dec3 = ConvSC(2*4*C_hid, 2*C_hid, stride=2,  transpose=True)
            self.dec2 = ConvSC(2*2*C_hid, C_hid, stride=1)
            self.dec1 = ConvSC(2*C_hid, C_hid, stride=1)
            self.readout_f = nn.Conv2d(C_hid, C_in, 1)
            self.readout_m = nn.Conv2d(C_hid, 1, 1)
        else:
            self.dec6 = ConvSC(8*C_hid, 8*C_hid, stride=1)
            self.dec5 = ConvSC(8*C_hid, 4*C_hid, stride=2, transpose=True)
            self.dec4 = ConvSC(4*C_hid, 4*C_hid,stride=1)
            self.dec3 = ConvSC(4*C_hid, 2*C_hid, stride=2,  transpose=True)
            self.dec2 = ConvSC(2*C_hid, C_hid, stride=1)
            self.dec1 = ConvSC(C_hid, C_hid, stride=1)
            self.readout_f = nn.Conv2d(C_hid, C_in, 1)
            self.readout_m = nn.Conv2d(C_hid, 1, 1)
    
    def forward(self,hid,skip=None):
        if self.unet:
            dec6 = self.dec6(hid)  # (128M + 128E), 32, 32 -> 128, 32, 32
            dec5 = self.dec5(torch.cat([dec6, skip[0]], dim=1)) # (128D + 128E), 32, 32 -> 64, 64, 64
            dec4 = self.dec4(torch.cat([dec5, skip[1]], dim=1)) # (64D + 64E), 64, 64   -> 64, 64, 64
            dec3 = self.dec3(torch.cat([dec4, skip[2]], dim=1)) # (64D + 64E), 64, 64   -> 32, 128, 128
            dec2 = self.dec2(torch.cat([dec3, skip[3]], dim=1)) # (32D + 32E), 128, 128 -> 16, 128, 128
            dec1 = self.dec1(torch.cat([dec2, skip[4]], dim=1)) # (16D + 16E), 128, 128 ->  4, 128, 128
            
            Y_f = self.readout_f(dec1)
            mask = torch.sigmoid( self.readout_m(dec1) )
        else:
            dec6 = self.dec6(hid)  # (128M + 128E), 32, 32 -> 128, 32, 32
            dec5 = self.dec5(dec6) # (128D + 128E), 32, 32 -> 64, 64, 64
            dec4 = self.dec4(dec5) # (64D + 64E), 64, 64   -> 64, 64, 64
            dec3 = self.dec3(dec4) # (64D + 64E), 64, 64   -> 32, 128, 128
            dec2 = self.dec2(dec3) # (32D + 32E), 128, 128 -> 16, 128, 128
            dec1 = self.dec1(dec2) # (16D + 16E), 128, 128 ->  4, 128, 128
            
            Y_f = self.readout_f(dec1)
            mask = torch.sigmoid( self.readout_m(dec1) )
        return Y_f,mask


class ConvUnet(nn.Module):
    def __init__(self, dataname,shape_in,C_hid=16, T_hid=4):
        super(ConvUnet,self).__init__()
        self.dataname = dataname
        T, C, H, W = shape_in

        if dataname == 'traffic':
            self.enc = EncoderShallow(C,C_hid)
            self.hid = Mid_Xnet(T+1,2*C_hid,T_hid,2*C_hid)
            self.dec = DecoderShallow(C,C_hid,unet=True)
        else:
            self.enc = EncoderDeep(C,C_hid)
            self.hid = Mid_Xnet(T+1,8*C_hid,T_hid,8*C_hid)
            self.dec = DecoderDeep(C,C_hid,unet=True)
    
    def forward(self,x_raw,b_mask):
        x_raw = torch.cat([x_raw,x_raw.mean(dim=1,keepdim=True)],dim=1)

        B,T,C,H,W = x_raw.shape
        x = x_raw.view(B*T,C,H,W)
        embed,skip = self.enc(x)
        _,C_,H_,W_ = embed.shape
        self.embed = embed

        z = embed.view(B,T*C_,H_,W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T,C_,H_,W_)

        Y_fb, mask = self.dec(hid,skip)
        Y_fb = Y_fb.reshape(B,T,C,H,W)
        Y_f = Y_fb[:,:-1]
        Y_b = Y_fb[:,-1:]
        mask = mask.reshape(B,T,1,H,W)[:,:-1]
        Y = Y_f*mask + Y_b*(1-mask)

        b_mask = b_mask.float()
        loss2 = torch.mean(b_mask*(Y_b - x_raw[:,0:1])**2)

        self.Y_f = Y_f
        self.Y_b = Y_b
        self.mask = mask
        return Y,loss2#[:,:-1,::]


if __name__ == '__main__':
    device = torch.device('cuda')
    T, C, H, W = 4,3,128,128
    x = torch.rand(8, T, C, H, W).to(device)
    model = ConvUnet('human',(T,C,H,W)).to(device)
    Y = model(x)
