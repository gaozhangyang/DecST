######################################### constrain moments ######################################################
import numpy as np
from numpy import *
from numpy.linalg import *
from scipy.special import factorial
from functools import reduce
import torch
import torch.nn as nn
from functools import reduce
import sys; sys.path.append('/usr/data/gzy/Weather_Forecast')

__all__ = ['M2K','K2M']

def _apply_axis_left_dot(x, mats):
    assert x.dim() == len(mats)+1
    sizex = x.size()
    k = x.dim()-1
    for i in range(k):
        x = tensordot(mats[k-i-1], x, dim=[1,k])
    x = x.permute([k,]+list(range(k))).contiguous()
    x = x.view(sizex)
    return x

def _apply_axis_right_dot(x, mats):
    assert x.dim() == len(mats)+1
    sizex = x.size()
    k = x.dim()-1
    x = x.permute(list(range(1,k+1))+[0,])
    for i in range(k):
        x = tensordot(x, mats[i], dim=[0,0])
    x = x.contiguous()
    x = x.view(sizex)
    return x

class _MK(nn.Module):
    def __init__(self, shape):
        super(_MK, self).__init__()
        self._size = torch.Size(shape)
        self._dim = len(shape)
        M = []
        invM = []
        assert len(shape) > 0
        j = 0
        for l in shape:
            M.append(zeros((l,l)))
            for i in range(l):
                M[-1][i] = ((arange(l)-(l-1)//2)**i)/factorial(i)
            invM.append(inv(M[-1]))
            self.register_buffer('_M'+str(j), torch.from_numpy(M[-1]))
            self.register_buffer('_invM'+str(j), torch.from_numpy(invM[-1]))
            j += 1

    @property
    def M(self):
        return list(self._buffers['_M'+str(j)] for j in range(self.dim()))
    @property
    def invM(self):
        return list(self._buffers['_invM'+str(j)] for j in range(self.dim()))

    def size(self):
        return self._size
    def dim(self):
        return self._dim
    def _packdim(self, x):
        assert x.dim() >= self.dim()
        if x.dim() == self.dim():
            x = x[newaxis,:]
        x = x.contiguous()
        x = x.view([-1,]+list(x.size()[-self.dim():]))
        return x

    def forward(self):
        pass

class M2K(_MK):
    """
    convert moment matrix to convolution kernel
    Arguments:
        shape (tuple of int): kernel shape
    Usage:
        m2k = M2K([5,5])
        m = torch.randn(5,5,dtype=torch.float64)
        k = m2k(m)
    """
    def __init__(self, shape):
        super(M2K, self).__init__(shape)
    def forward(self, m):
        """
        m (Tensor): torch.size=[...,*self.shape]
        """
        sizem = m.size()
        m = self._packdim(m)
        m = _apply_axis_left_dot(m, self.invM)
        m = m.view(sizem)
        return m

class K2M(_MK):
    """
    convert convolution kernel to moment matrix
    Arguments:
        shape (tuple of int): kernel shape
    Usage:
        k2m = K2M([5,5])
        k = torch.randn(5,5,dtype=torch.float64)
        m = k2m(k)
    """
    def __init__(self, shape):
        super(K2M, self).__init__(shape)
    def forward(self, k):
        """
        k (Tensor): torch.size=[...,*self.shape]
        """
        sizek = k.size()
        k = self._packdim(k)
        k = _apply_axis_left_dot(k, self.M)
        k = k.view(sizek)
        return k


    
def tensordot(a,b,dim):
    """
    tensordot in PyTorch, see numpy.tensordot?
    """
    l = lambda x,y:x*y
    if isinstance(dim,int):
        a = a.contiguous()
        b = b.contiguous()
        sizea = a.size()
        sizeb = b.size()
        sizea0 = sizea[:-dim]
        sizea1 = sizea[-dim:]
        sizeb0 = sizeb[:dim]
        sizeb1 = sizeb[dim:]
        N = reduce(l, sizea1, 1)
        assert reduce(l, sizeb0, 1) == N
    else:
        adims = dim[0]
        bdims = dim[1]
        adims = [adims,] if isinstance(adims, int) else adims
        bdims = [bdims,] if isinstance(bdims, int) else bdims
        adims_ = set(range(a.dim())).difference(set(adims))
        adims_ = list(adims_)
        adims_.sort()
        perma = adims_+adims
        bdims_ = set(range(b.dim())).difference(set(bdims))
        bdims_ = list(bdims_)
        bdims_.sort()
        permb = bdims+bdims_
        a = a.permute(*perma).contiguous()
        b = b.permute(*permb).contiguous()

        sizea = a.size()
        sizeb = b.size()
        sizea0 = sizea[:-len(adims)]
        sizea1 = sizea[-len(adims):]
        sizeb0 = sizeb[:len(bdims)]
        sizeb1 = sizeb[len(bdims):]
        N = reduce(l, sizea1, 1)
        assert reduce(l, sizeb0, 1) == N
    a = a.view([-1,N])
    b = b.view([N,-1])
    c = a@b
    return c.view(sizea0+sizeb1)


############################################ model #######################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PhyCell_Cell(nn.Module):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.input_dim  = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.F = nn.Sequential()
        self.F.add_module('bn1',nn.GroupNorm( 4 ,input_dim))          
        self.F.add_module('conv1', nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size, stride=(1,1), padding=self.padding))  
        #self.F.add_module('f_act1', nn.LeakyReLU(negative_slope=0.1))        
        self.F.add_module('conv2', nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1,1), stride=(1,1), padding=(0,0)))

        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                              out_channels= self.input_dim,
                              kernel_size=(3,3),
                              padding=(1,1), bias=self.bias)

    def forward(self, x, hidden): # x [batch_size, hidden_dim, height, width]      
        hidden_tilde = hidden + self.F(hidden)        # prediction
        
        combined = torch.cat([x, hidden_tilde], dim=1)  # concatenate along channel axis
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        
        next_hidden = hidden_tilde + K * (x-hidden_tilde)   # correction , Haddamard product     
        return next_hidden

   
class PhyCell(nn.Module):
    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size, device):
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []  
        self.device = device
             
        cell_list = []
        for i in range(0, self.n_layers):
        #    cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]

            cell_list.append(PhyCell_Cell(input_dim=input_dim,
                                          F_hidden_dim=self.F_hidden_dims[i],
                                          kernel_size=self.kernel_size))                                     
        self.cell_list = nn.ModuleList(cell_list)
        
       
    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]    
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size) # init Hidden at each forward start
              
        for j,cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                self.H[j] = cell(input_, self.H[j])
            else:
                self.H[j] = cell(self.H[j-1],self.H[j])
        
        return self.H , self.H 
    
    def initHidden(self,batch_size):
        self.H = [] 
        for i in range(self.n_layers):
            self.H.append( torch.zeros(batch_size, self.input_dim, self.input_shape[0], self.input_shape[1]).to(self.device) )

    def setHidden(self, H):
        self.H = H
  
   
class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):              
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM_Cell, self).__init__()
        
        self.height, self.width = input_shape
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)
                 
    # we implement LSTM that process only one timestep 
    def forward(self,x, hidden): # x [batch, hidden_dim, width, height]          
        h_cur, c_cur = hidden
        
        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


    
    
class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size,device):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [],[]   
        self.device = device
        
        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
            print('layer ',i,'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(ConvLSTM_Cell(input_shape=self.input_shape,
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dims[i],
                                          kernel_size=self.kernel_size))                                     
        self.cell_list = nn.ModuleList(cell_list)
        
       
    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]    
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size) # init Hidden at each forward start
              
        for j,cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j],self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j-1],(self.H[j],self.C[j]))
        
        return (self.H,self.C) , self.H   # (hidden, output)
    
    def initHidden(self,batch_size):
        self.H, self.C = [],[]  
        for i in range(self.n_layers):
            self.H.append( torch.zeros(batch_size,self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device) )
            self.C.append( torch.zeros(batch_size,self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device) )
    
    def setHidden(self, hidden):
        H,C = hidden
        self.H, self.C = H,C
 

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3,3), stride=stride, padding=1),
                nn.GroupNorm(4,nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

        
class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if (stride ==2):
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channels=nin,out_channels=nout,kernel_size=(3,3), stride=stride,padding=1,output_padding=output_padding),
                nn.GroupNorm(4,nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)
     

class image_encoder(nn.Module):
    def __init__(self, nc=1):
        super(image_encoder, self).__init__()
        nf = 16
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf, stride=1) # (nf) x 64 x 64
        self.c2 = dcgan_conv(nf, nf*2, stride=1) # (nf) x 64 x 64
        self.c3 = dcgan_conv(nf*2, nf*4, stride=2) # (2*nf) x 32 x 32
        self.c4 = dcgan_conv(nf*4, nf*4, stride=1) # (2*nf) x 32 x 32
        self.c5 = dcgan_conv(nf*4, nf*8, stride=2) # (4*nf) x 16 x 16
        self.c6 = dcgan_conv(nf*8, nf*8, stride=1) # (4*nf) x 16 x 16          

    def forward(self, input):
        h1 = self.c1(input)  # (nf/2) x 64 x 64
        h2 = self.c2(h1)     # (nf) x 64 x 64
        h3 = self.c3(h2)     # (2*nf) x 32 x 32
        h4 = self.c4(h3)     # (2*nf) x 32 x 32
        h5 = self.c5(h4)     # (4*nf) x 16 x 16
        h6 = self.c6(h5)     # (4*nf) x 16 x 16          
        return h6, [h1, h2, h3, h4, h5, h6]


class image_decoder(nn.Module):
    def __init__(self, nc=1):
        super(image_decoder, self).__init__()
        nf = 16
        self.upc1 = dcgan_upconv(nf*8*2, nf*8, stride=1) #(nf*4) x 16 x 16
        self.upc2 = dcgan_upconv(nf*8*2, nf*4, stride=2) #(nf*2) x 32 x 32
        self.upc3 = dcgan_upconv(nf*4*2, nf*4, stride=1) #(nf*2) x 32 x 32
        self.upc4 = dcgan_upconv(nf*4*2, nf*2, stride=2)   #(nf) x 64 x 64
        self.upc5 = dcgan_upconv(nf*2*2, nf, stride=1)   #(nf/2) x 64 x 64
        self.upc6 = nn.ConvTranspose2d(in_channels=nf*2,out_channels=nc,kernel_size=(3,3),stride=1,padding=1)  #(nc) x 64 x 64

    def forward(self, input):
        vec, skip = input    # vec: (4*nf) x 16 x 16          
        [h1, h2, h3, h4, h5, h6] = skip
        d1 = self.upc1(torch.cat([vec, h6], dim=1))  #(nf*4) x 16 x 16
        d2 = self.upc2(torch.cat([d1, h5], dim=1))   #(nf*2) x 32 x 32
        d3 = self.upc3(torch.cat([d2, h4], dim=1))   #(nf*2) x 32 x 32
        d4 = self.upc4(torch.cat([d3, h3], dim=1))   #(nf) x 64 x 64
        d5 = self.upc5(torch.cat([d4, h2], dim=1))   #(nf/2) x 64 x 64
        d6 = self.upc6(torch.cat([d5, h1], dim=1))   #(nc) x 64 x 64
        return d6
        

class EncoderRNN(torch.nn.Module):
    def __init__(self,phycell,convlstm, device):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.image_cnn_enc = image_encoder(nc=3).to(device) # image encoder 64x64x1 -> 16x16x64
        self.image_cnn_dec = image_decoder(nc=3).to(device) # image decoder 16x16x64 -> 64x64x1 
        
        self.phycell = phycell.to(device)
        self.convlstm = convlstm.to(device)

        
    def forward(self, input, first_timestep=False, decoding=False):
        if decoding:  # input=None in decoding phase
            output_phys = None
        else:
            output_phys,skip = self.image_cnn_enc(input)
        output_conv,skip = self.image_cnn_enc(input)     

        hidden1, output1 = self.phycell(output_phys, first_timestep)
        hidden2, output2 = self.convlstm(output_conv, first_timestep)

        out_phys = torch.sigmoid(self.image_cnn_dec([output1[-1],skip])) # partial reconstructions for vizualization
        out_conv = torch.sigmoid(self.image_cnn_dec([output2[-1],skip]))

        concat = output1[-1]+output2[-1]
        output_image = torch.sigmoid( self.image_cnn_dec([concat,skip]) )
        return out_phys, hidden1, output_image, out_phys, out_conv

##################################################### API #####################################
import os
from tqdm import tqdm

class PhyDNet:
    def __init__(self, args):
        super(PhyDNet, self).__init__()
        device = 'cuda:0'
        self.device = device
        phycell =  PhyCell(input_shape=(32,32), input_dim=128, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device) 
        convlstm =  ConvLSTM(input_shape=(32,32), input_dim=128, hidden_dims=[256,256,128], n_layers=3, kernel_size=(3,3), device=device)   
        self.model = EncoderRNN(phycell, convlstm, device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0001)
        self.criterion = nn.MSELoss()

        self.path = args.res_dir+'/{}'.format(args.ex_name)
        self.folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.constraints = torch.zeros((49,7,7)).to(self.device)
        ind = 0
        for i in range(0,7):
            for j in range(0,7):
                self.constraints[ind,i,j] = 1
                ind +=1 

    def train(self,train_loader, epoch): 
        self.model.train()
           
        
        # teacher_forcing_ratio = np.maximum(0 , 1 - epoch * 0.01)
        teacher_forcing_ratio = 0

        train_pbar = tqdm(train_loader)
        for input_tensor, target_tensor,_ in train_pbar:
            self.optimizer.zero_grad()
            # input_tensor : torch.Size([batch_size, input_length, 1, 64, 64])
            input_tensor = input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)
            input_length  = input_tensor.size(1)
            target_length = target_tensor.size(1)
            loss = 0
            for ei in range(input_length-1): 
                model_output, model_hidden, output_image,_,_ = self.model(input_tensor[:,ei,:,:,:], (ei==0) )
                loss += self.criterion(output_image,input_tensor[:,ei+1,:,:,:])

            decoder_input = input_tensor[:,-1,:,:,:] # first decoder input = last image of input sequence
            
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False 
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, output_image,_,_ = self.model(decoder_input)
                    target = target_tensor[:,di,:,:,:]
                    loss += self.criterion(output_image,target)
                    decoder_input = target 
            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, output_image,_,_ = self.model(decoder_input)
                    decoder_input = output_image
                    target = target_tensor[:,di,:,:,:]
                    loss += self.criterion(output_image, target)
            mse_loss = loss
        
            # Moment Regularisation  model.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
            k2m = K2M([7,7]).to(self.device)
            for b in range(0,self.model.phycell.cell_list[0].input_dim):
                filters = self.model.phycell.cell_list[0].F.conv1.weight[:,b,:,:] # (nb_filters,7,7)
                
                m = k2m(filters.double()) 
                m  = m.float()   
                loss += self.criterion(m, self.constraints) # constrains is a precomputed matrix   
            loss.backward()
            self.optimizer.step()
        return mse_loss.item()
    

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

                input_length = input_tensor.size()[1]
                target_length = target_tensor.size()[1]

                for ei in range(input_length-1):
                    model_output, model_hidden, _,_,_  = self.model(input_tensor[:,ei,:,:,:], (ei==0))

                decoder_input = input_tensor[:,-1,:,:,:] # first decoder input= last image of input sequence
                predictions = []

                for di in range(target_length):
                    decoder_output, decoder_hidden, output_image,_,_ = self.model(decoder_input, False, False)
                    decoder_input = output_image
                    predictions.append(output_image.cpu())

                target = target_tensor.cpu().numpy()
                predictions =  np.stack(predictions) # for MM: (10, batch_size, 1, 64, 64)
                predictions = predictions.swapaxes(0,1)  # (batch_size,10, 1, 64, 64)
                preds.append(predictions)
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

                input_length = input_tensor.size()[1]
                target_length = target_tensor.size()[1]

                for ei in range(input_length-1):
                    model_output, model_hidden, _,_,_  = self.model(input_tensor[:,ei,:,:,:], (ei==0))

                decoder_input = input_tensor[:,-1,:,:,:] # first decoder input= last image of input sequence
                predictions = []

                for di in range(target_length):
                    decoder_output, decoder_hidden, output_image,_,_ = self.model(decoder_input, False, False)
                    decoder_input = output_image
                    predictions.append(output_image.cpu())

                target = target_tensor.cpu().numpy()
                predictions =  np.stack(predictions) # for MM: (10, batch_size, 1, 64, 64)
                predictions = predictions.swapaxes(0,1)  # (batch_size,10, 1, 64, 64)
                preds.append(predictions)
                trues.append(target)

                if number>=1000:
                    break

        preds = np.concatenate(preds,axis=0)
        trues = np.concatenate(trues,axis=0)
        np.save(self.folder_path+'true.npy', trues)
        np.save(self.folder_path+'pred.npy', preds)

        import sys; sys.path.append('/usr/data/gzy/Weather_Forecast')
        from API.metrics import metric
        mae, mse, rmse, mape, mspe = metric(preds, trues,test_loader.dataset.mean,test_loader.dataset.std)
        return mse, mae






