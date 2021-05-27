import sys; sys.path.append('..')
from ex_archi_search.conv import BasicConv1d, BasicConv2d, Conv3D_,GroupConv2d
import torch.nn as nn
import torch

class Conv3D(nn.Module):
    def __init__(self,in_shape,out_shape,reverse=False):
        super(Conv3D,self).__init__()
        self.conv = Conv3D_(in_shape, out_shape, act_norm=True)
    
    def forward(self,x):
        y = self.conv(x)
        return y

class ConvTC(nn.Module):
    def __init__(self, in_shape, out_shape, reverse=False):
        super(ConvTC, self).__init__()
        T, C, H, W = in_shape
        T_, C_, H_, W_ = out_shape
        transpose = True if T_ > T else False
        stride = int(max(T / T_, T_ / T))
        self.conv = BasicConv1d(C, C_, kernel_size=3, stride=stride, padding=1, transpose=transpose, act_norm=True)
    
    def forward(self, x):
        B,T,C,H,W = x.shape
        z = x.permute(0,3,4,2,1).reshape(B*H*W,C,T)
        y = self.conv(z)
        _, C_, T_ = y.shape
        y = y.reshape(B,H,W,C_,T_).permute(0,4,3,1,2).contiguous()
        return y

class ConvSC(nn.Module):
    def __init__(self, in_shape, out_shape, reverse=False):
        super(ConvSC,self).__init__()
        T, C, H, W = in_shape
        T_, C_, H_, W_ = out_shape
        transpose = True if W_ > W else False
        stride = int(max(W / W_, W_ / W))
        self.conv = BasicConv2d(C, C_, kernel_size=3, stride=stride, padding=1, transpose=transpose, act_norm=True)
    
    def forward(self,x):
        B,T,C,H,W = x.shape
        z = x.reshape(B*T,C,H,W)
        y = self.conv(z)
        _,C_,H_,W_ = y.shape
        y = y.reshape(B,T,C_,H_,W_)
        return y

class Mid_Unet(nn.Module):
    def __init__(self, hid_module, in_shape, hidden_shape):
        super(Mid_Unet, self).__init__()
        dec_shape = list(in_shape)
        dec_shape[1] *= 2
        dec_shape = tuple(dec_shape)
        self.enc1 = hid_module(in_shape, hidden_shape)
        self.enc2 = hid_module(hidden_shape, hidden_shape)
        self.enc3 = hid_module(hidden_shape, hidden_shape)
        self.dec3 = hid_module(hidden_shape, hidden_shape)
        self.dec2 = hid_module(dec_shape, hidden_shape)
        self.dec1 = hid_module(dec_shape, hidden_shape)

    def forward(self, x):
        enc1 = self.enc1(x) 
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        dec3 = self.dec3(enc3)
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=2))
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=2))
        return dec1

class gInception_ST(nn.Module):
    def __init__(self, T_in,C_in,T_out,C_out,gConv,Incep):        
        super(gInception_ST, self).__init__()
        self.gConv=gConv
        self.Incep=Incep

        ksize=[3,5,7,11]
        stride=[1,1,1,1]
        self.T_in, self.C_in, self.T_out, self.C_out = T_in,C_in,T_out,C_out
        in_channels, out_channels = T_in*C_in, T_out*C_out
        # groups = C_in if (out_channels // 2) % C_in==0 else C_in//2
        groups = 8
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        if Incep:
            self.conv_3x3 = GroupConv2d(out_channels, out_channels, kernel_size=ksize[0], stride=stride[0],padding=ksize[0]//2, groups=groups, act_norm=True,gConv=gConv)
            self.conv_5x5 = GroupConv2d(out_channels, out_channels, kernel_size=ksize[1], stride=stride[1], padding=ksize[1]//2,groups=groups, act_norm=True,gConv=gConv)
            self.conv_7x7 = GroupConv2d(out_channels, out_channels, kernel_size=ksize[2], stride=stride[2], padding=ksize[2]//2,groups=groups, act_norm=True,gConv=gConv)   
            self.conv_11x11 = GroupConv2d(out_channels, out_channels, kernel_size=ksize[3], stride=stride[3], padding=ksize[3]//2,groups=groups, act_norm=True,gConv=gConv)
        else:
            self.conv_3x3 = GroupConv2d(out_channels, out_channels, kernel_size=ksize[0], stride=stride[0],padding=ksize[0]//2, groups=groups, act_norm=True,gConv=gConv)

    def forward(self, x):
        x = self.conv1(x)
        if self.Incep:
            y = self.conv_3x3(x) + self.conv_5x5(x) + self.conv_7x7(x)+self.conv_11x11(x)
        else:
            y = self.conv_3x3(x)
        return y

class Mid_Xnet(nn.Module):
    def __init__(self, T_in,C_in,T_hid,C_hid,XNet,gConv,Incep):
        super(Mid_Xnet, self).__init__()
        self.T_in ,self.C_in ,self.T_hid ,self.C_hid = T_in,C_in,T_hid,C_hid
        self.XNet=XNet
        self.gConv=gConv
        self.Incep=Incep

        self.enc1 = gInception_ST(T_in,C_in,T_hid,C_hid,gConv,Incep)
        self.enc2 = gInception_ST(T_hid,C_hid,T_hid,C_hid,gConv,Incep)
        self.enc3 = gInception_ST(T_hid,C_hid,T_hid,C_hid,gConv,Incep)
        self.dec3 = gInception_ST(2*T_hid,C_hid,T_hid,C_hid, gConv,Incep)
        self.dec2 = gInception_ST(2*T_hid,C_hid,T_hid,C_hid, gConv,Incep)
        self.dec1 = gInception_ST(2*T_hid,C_hid,T_in,C_in, gConv,Incep)

    def forward(self, x):
        B,T,C,H,W=x.shape
        x = x.view(B,T*C,H,W)
        enc1 = self.enc1(x) #[64, 160, 16, 16]
        enc2 = self.enc2(enc1) #[64, 128, 16, 16]
        enc3 = self.enc3(enc2) #[64, 128, 16, 16]

        if self.XNet:
            dec3 = self.dec3(torch.cat([enc3, enc2], dim=1)) #[64, 128, 16, 16]
            dec2 = self.dec2(torch.cat([dec3, enc1+enc2+enc3], dim=1)) #[64, 128, 16, 16]
            dec1 = self.dec1(torch.cat([dec2, enc1+enc2], dim=1)) #[64, 128, 16, 16]
        else:
            dec3 = self.dec3(torch.cat([enc3, enc3], dim=1)) #[64, 128, 16, 16]
            dec2 = self.dec2(torch.cat([dec3, enc2], dim=1)) #[64, 128, 16, 16]
            dec1 = self.dec1(torch.cat([dec2, enc1], dim=1)) #[64, 128, 16, 16]

        dec1 = dec1.view(B,T,C,H,W)
        return dec1

def get_binary_mask(x):
    error = torch.abs(x-x[:,0:1]).sum(dim=2)
    binary_mask = error<0.05
    y = binary_mask[:,0]
    for t in range(1,binary_mask.shape[1]):
        y = y&binary_mask[:,t]
    return y

conv_map = {
                     # END    # Hid
            'S_ST':  (ConvSC, Conv3D),
            'ST_S':  (Conv3D, ConvSC),
            'ST_ST': (Conv3D, Conv3D),
            'gXNet': (ConvSC, Mid_Xnet),
            }

class ConvUnet(nn.Module):
    def __init__(self, method,shape_in,UNet=True,XNet=True,gConv=True,Incep=True):
        super(ConvUnet,self).__init__()
        self.method = method
        T, C, H, W = shape_in
        c_hid = 16
        shape_enc = [   [shape_in,                  (T, c_hid, H, W)], 
                        [(T, c_hid, H, W),          (T, c_hid*2, H//2, W//2)], 
                        [(T, c_hid*2, H//2, W//2),  (T, c_hid*2, H//2, W//2)],
                    ]
        
        shape_hid = [   [(T, c_hid*2, H//2, W//2),  (T, c_hid*2, H//2, W//2)],
                        [(T, c_hid*2, H//2, W//2),  (T, c_hid*2, H//2, W//2)],
                        [(T, c_hid*2, H//2, W//2),  (T, c_hid*2, H//2, W//2)],
                        [(T, c_hid*2, H//2, W//2),  (T, c_hid*2, H//2, W//2)],
                        [(T, c_hid*2, H//2, W//2),  (T, c_hid*2, H//2, W//2)],
                        [(T, c_hid*2, H//2, W//2),  (T, c_hid*2, H//2, W//2)],
                    ]

        shape_dec_u = [
                        [(T, c_hid*2, H//2, W//2),  (T, c_hid*2, H//2, W//2)],
                        [(T, c_hid*4, H//2, W//2),  (T, c_hid, H, W)],
                        [(T, c_hid*2, H, W),        (T, c_hid, H, W)]
                    ]
        
        shape_dec = [
                        [(T, c_hid*2, H//2, W//2),  (T, c_hid*2, H//2, W//2)],
                        [(T, c_hid*2, H//2, W//2),  (T, c_hid, H, W)],
                        [(T, c_hid, H, W),          (T, c_hid, H, W)]
                    ]
    
        end, hid = conv_map[method]

        self.UNet = UNet
        self.enc1 = end(shape_enc[0][0],shape_enc[0][1])
        self.enc2 = end(shape_enc[1][0],shape_enc[1][1])
        self.enc3 = end(shape_enc[2][0],shape_enc[2][1])

        if hid==Mid_Xnet:
            self.hid = Mid_Xnet(T,c_hid*2,T,c_hid*2,XNet,gConv,Incep)
        else:
            self.hid = Mid_Unet(hid, shape_hid[0][0], shape_hid[0][1])

        if self.UNet:
            self.dec3 = end(shape_dec_u[0][0],shape_dec_u[0][1])
            self.dec2 = end(shape_dec_u[1][0],shape_dec_u[1][1])
            self.dec1 = end(shape_dec_u[2][0],shape_dec_u[2][1])
        else:
            self.dec3 = end(shape_dec[0][0],shape_dec[0][1])
            self.dec2 = end(shape_dec[1][0],shape_dec[1][1])
            self.dec1 = end(shape_dec[2][0],shape_dec[2][1])
    
    def forward(self,x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        hid = self.hid(enc3)

        if self.UNet:
            dec3 = self.dec3(hid)
            dec2 = self.dec2(torch.cat([dec3, enc2], dim=2))
            dec1 = self.dec1(torch.cat([dec2, enc1], dim=2))
        else:
            dec3 = self.dec3(hid)
            dec2 = self.dec2(dec3)
            dec1 = self.dec1(dec2)

        B,T,C,H,W = dec1.shape
        dec1 = dec1.reshape(B*T,C,H,W)

        Y = self.readout(dec1)
        _,C,H,W = Y.shape
        Y = Y.reshape(B,T,C,H,W)
        return Y,0


class DecST(nn.Module):
    def __init__(self, shape_in):
        super(DecST,self).__init__()
        T, C, H, W = shape_in
        T = T+1
        c_hid = 16
        shape_enc = [   [shape_in,                  (T, c_hid, H, W)], 
                        [(T, c_hid, H, W),          (T, c_hid*2, H//2, W//2)], 
                        [(T, c_hid*2, H//2, W//2),  (T, c_hid*2, H//2, W//2)],
                    ]
        
        shape_hid = [   [(T, c_hid*2, H//2, W//2),  (T, c_hid*2, H//2, W//2)],
                        [(T, c_hid*2, H//2, W//2),  (T, c_hid*2, H//2, W//2)],
                        [(T, c_hid*2, H//2, W//2),  (T, c_hid*2, H//2, W//2)],
                        [(T, c_hid*2, H//2, W//2),  (T, c_hid*2, H//2, W//2)],
                        [(T, c_hid*2, H//2, W//2),  (T, c_hid*2, H//2, W//2)],
                        [(T, c_hid*2, H//2, W//2),  (T, c_hid*2, H//2, W//2)],
                    ]

        shape_dec_u = [
                        [(T, c_hid*2, H//2, W//2),  (T, c_hid*2, H//2, W//2)],
                        [(T, c_hid*4, H//2, W//2),  (T, c_hid, H, W)],
                        [(T, c_hid*2, H, W),        (T, c_hid, H, W)]
                    ]
        
    
        end, hid = ConvSC, Mid_Xnet

        self.enc1 = end(shape_enc[0][0],shape_enc[0][1])
        self.enc2 = end(shape_enc[1][0],shape_enc[1][1])
        self.enc3 = end(shape_enc[2][0],shape_enc[2][1])
        self.hid = Mid_Xnet(T,c_hid*2,T,c_hid*2,XNet=True,gConv=True,Incep=True)
        self.dec3 = end(shape_dec_u[0][0],shape_dec_u[0][1])
        self.dec2 = end(shape_dec_u[1][0],shape_dec_u[1][1])
        self.dec1 = end(shape_dec_u[2][0],shape_dec_u[2][1])
        self.readout_f = nn.Conv2d(c_hid, C, 1)
        self.readout_m = nn.Conv2d(c_hid, 1, 1)

    
    def forward(self,x):
        b_mask = get_binary_mask(x)
        x = torch.cat([x,x.mean(dim=1,keepdim=True)],dim=1)
        B,T,C,H,W = x.shape

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        hid = self.hid(enc3)

        dec3 = self.dec3(hid)
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=2))
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=2))
        B,T,C,H,W = dec1.shape
        dec1 = dec1.reshape(B*T,C,H,W)

        Y_fb = self.readout_f(dec1)
        _,C,H,W = Y_fb.shape
        Y_fb = Y_fb.reshape(B,T,C,H,W)
        mask = torch.sigmoid(self.readout_m(dec1))
        mask = mask.reshape(B,T,1,H,W)[:,:-1]

        Y_f = Y_fb[:,:-1]
        Y_b = Y_fb[:,-1:]
        # mask = mask.reshape(B,T,1,H,W)[:,:-1]
        Y = Y_f*mask + Y_b*(1-mask)

        b_mask = b_mask.view(B,1,1,H,W).float()
        loss2 = torch.mean(b_mask*(Y_b - x[:,0:1])**2)
        self.Y_f = Y_f
        self.Y_b = Y_b
        self.mask = mask
        return Y,loss2


if __name__ == '__main__':
    device = torch.device('cuda')
    T, C, H, W = 4,2,32,32
    x = torch.rand(8, T, C, H, W).to(device)
    for k, v in conv_map.items():
        print(k)
        model = ConvUnet(k,(T, C, H, W),UNet=True,XNet=True,gConv=True,Incep=False).to(device)
        y = model(x)
