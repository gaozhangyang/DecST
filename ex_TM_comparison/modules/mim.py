import torch
import torch.nn as nn
from .rnn_cell import MIMN as mimn
from .rnn_cell import MIMBlock as mimblock
from .rnn_cell import MIM_SpatioTemporalLSTMCell as stlstm
from tqdm import tqdm
import numpy as np
from .utils import reshape_patch, reshape_patch_back

# Cite from https://github.com/coolsunxu/MIM_Pytorch, Thanks!
class MIM_model(nn.Module):  
    def __init__(self, params):
        super(MIM_model, self).__init__()
        self.num_layers = params.num_layers
        self.num_hidden = params.num_hidden
        self.filter_size = params.filter_size
        self.total_length = params.total_length
        self.input_length = params.input_length
        self.tln = True

        self.stlstm_layer = nn.ModuleList() 
        self.stlstm_layer_diff = nn.ModuleList()
        self.cell_state = [] 
        self.hidden_state = [] 
        self.cell_state_diff = [] 
        self.hidden_state_diff = [] 
        # self.shape = [params.batch_size, 3, 1, 128, 128]
        self.shape = [params.batch_size, self.input_length, params.img_channel, params.img_width, params.img_width] 
        self.output_channels = self.shape[-3] 
		
        for i in range(self.num_layers): 
            if i == 0:
                num_hidden_in = self.num_hidden[self.num_layers - 1] 
            else:
                num_hidden_in = self.num_hidden[i - 1] 
            if i < 1: 
                new_stlstm_layer = stlstm('stlstm_' + str(i + 1),
							  self.filter_size,
							  num_hidden_in,
							  self.num_hidden[i],
							  self.shape,
							  self.output_channels,
							  tln=self.tln)
            else: 
                new_stlstm_layer = mimblock('stlstm_' + str(i + 1),
								self.filter_size,
								num_hidden_in,
								self.num_hidden[i],
								self.shape,
								self.num_hidden[i-1],
								tln=self.tln)
            self.stlstm_layer.append(new_stlstm_layer) 
            self.cell_state.append(None) 
            self.hidden_state.append(None)

        for i in range(self.num_layers - 1): 
            new_stlstm_layer = mimn('stlstm_diff' + str(i + 1),
								self.filter_size,
								self.num_hidden[i + 1],
								self.shape,
								tln=self.tln)
            self.stlstm_layer_diff.append(new_stlstm_layer)
            self.cell_state_diff.append(None)
            self.hidden_state_diff.append(None)

        self.st_memory = None 
		
        self.x_gen = nn.Conv2d(self.num_hidden[self.num_layers - 1],
				 self.output_channels,1,1,padding=0
				 )


    def forward(self, images, schedual_sampling_bool):
        gen_images = []
        batch_size = images.shape[0]
        for time_step in range(self.total_length): 			
            if time_step < self.input_length:
                x_gen = images[:,time_step] 
            else:
                x_gen = schedual_sampling_bool[:batch_size,time_step-self.input_length]*images[:,time_step] + \
                        (1-schedual_sampling_bool[:batch_size,time_step-self.input_length])*x_gen
                        
            preh = self.hidden_state[0]
            hidden_state, cell_state, st_memory = self.stlstm_layer[0]( 
                x_gen, self.hidden_state[0], self.cell_state[0], self.st_memory, batch_size)
            self.hidden_state[0] = hidden_state.clone().detach()
            self.cell_state[0] = cell_state.clone().detach()
            self.st_memory = st_memory.clone().detach()
            
            for i in range(1, self.num_layers):
                if time_step > 0:  
                    if i == 1:
                        hidden_state, cell_state = self.stlstm_layer_diff[i - 1](
                            self.hidden_state[i - 1] - preh, self.hidden_state_diff[i - 1], self.cell_state_diff[i - 1], batch_size)
                    else:
                        hidden_state, cell_state = self.stlstm_layer_diff[i - 1](
                            self.hidden_state_diff[i - 2], self.hidden_state_diff[i - 1], self.cell_state_diff[i - 1], batch_size)
                    self.hidden_state_diff[i - 1] = hidden_state.clone().detach()
                    self.cell_state_diff[i - 1] = cell_state.clone().detach()
                else:
                    self.stlstm_layer_diff[i - 1](torch.zeros_like(self.hidden_state[i - 1]), None, None, batch_size)
                preh = self.hidden_state[i]
                hidden_state, cell_state, st_memory = self.stlstm_layer[i]( 
                    self.hidden_state[i - 1], self.hidden_state_diff[i - 1], self.hidden_state[i], self.cell_state[i], self.st_memory, batch_size)
                self.hidden_state[i] = hidden_state.clone().detach()
                self.cell_state[i] = cell_state.clone().detach()
                self.st_memory = st_memory.clone().detach()

            x_gen = self.x_gen(self.hidden_state[self.num_layers - 1])
            gen_images.append(x_gen)
        
        self._clean()
        gen_images = torch.stack(gen_images, dim=1) 
        loss_fn = nn.MSELoss()
        loss = loss_fn(gen_images, images)
        return gen_images, loss


    def _clean(self):
        self.cell_state = [] 
        self.hidden_state = [] 
        self.cell_state_diff = [] 
        self.hidden_state_diff = [] 
        for i in range(self.num_layers): 
            self.cell_state.append(None)
            self.hidden_state.append(None)
            self.cell_state_diff.append(None)
            self.hidden_state_diff.append(None) 
        self.st_memory = None 
        for i, module in enumerate(self.stlstm_layer):
            if i > 0:
                module.convlstm_c = None

# from .basic_algo import Basic_algo
class MIM:
    def __init__(self, params):
        config = params.__dict__
        config.update({
            'lr': 0.001,
            'device': torch.device('cuda:0'),
            'num_hidden': [64,64,64,64],
            'num_layers': 4,
            'patch_size': 1,
            'img_channel': 3,
            'img_width': 128,
            'filter_size': 5,
            'stride': 1,
            'layer_norm': True,
            'total_length': 8,
            'reverse_scheduled_sampling': 0,
            'input_length': 4,
            'max_iterations': 80000,
            'sampling_start_value': 1.0,
            'sampling_stop_iter': 50000,
            'sampling_changing_rate': 0.00002,
            'scheduled_sampling': 1,
            'batch_size': 64
        })
        self.device = params.device
        self.model = MIM_model(params).to(self.device)
        # Basic_algo.__init__(self, model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)
        self.criterion = torch.nn.MSELoss()
        self.iterations = 0
        self.max_iterations = params.max_iterations
        self.eta = params.sampling_start_value
        self.params = params

    def schedule_sampling(self, eta, itr, args):
        zeros = np.zeros((args.batch_size,
                        # args.total_length - args.input_length - 1,
                        args.total_length - args.input_length,
                        args.img_width // args.patch_size,
                        args.img_width // args.patch_size,
                        args.patch_size ** 2 * args.img_channel))
        if not args.scheduled_sampling:
            return 0.0, zeros

        if itr < args.sampling_stop_iter:
            eta -= args.sampling_changing_rate
        else:
            eta = 0.0
        # random_flip = np.random.random_sample(
        #     (args.batch_size, args.total_length - args.input_length - 1))
        random_flip = np.random.random_sample(
            (args.batch_size, args.total_length - args.input_length))
        true_token = (random_flip < eta)
        ones = np.ones((args.img_width // args.patch_size,
                        args.img_width // args.patch_size,
                        args.patch_size ** 2 * args.img_channel))
        zeros = np.zeros((args.img_width // args.patch_size,
                        args.img_width // args.patch_size,
                        args.patch_size ** 2 * args.img_channel))
        real_input_flag = []
        for i in range(args.batch_size):
            # for j in range(args.total_length - args.input_length - 1):
            for j in range(args.total_length - args.input_length):
                if true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
        real_input_flag = np.array(real_input_flag)
        real_input_flag = np.reshape(real_input_flag,
                                    (args.batch_size,
                                    # args.total_length - args.input_length - 1,
                                    args.total_length - args.input_length,
                                    args.img_width // args.patch_size,
                                    args.img_width // args.patch_size,
                                    args.patch_size ** 2 * args.img_channel))
        return eta, real_input_flag

    def train(self, train_loader, epoch): 
        self.model.train()
        train_pbar = tqdm(train_loader)
        mse_loss = []
        for i, (batch_x, batch_y,_) in enumerate(train_pbar):
            self.iterations += 1
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            # train model
            batch_x = torch.cat([batch_x, batch_y], dim=1)
            self.eta, real_input_flag = self.schedule_sampling(self.eta, self.iterations, self.params)
            real_input_flag = torch.FloatTensor(real_input_flag).to(self.device)
            self.optimizer.zero_grad()
            pred, loss = self.model(batch_x, real_input_flag.permute(0, 1, 4, 2, 3).contiguous())
            loss.backward(retain_graph=True)
            self.optimizer.step()
            # trian model
            train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
            mse_loss.append(loss.item())
        mse_loss = np.average(mse_loss)
        return mse_loss

    def evaluate(self, val_loader):
        self.model.eval()
        real_input_flag = np.zeros(
                                (self.params.batch_size,
                                # self.params.total_length - self.params.input_length - 1,
                                self.params.total_length - self.params.input_length,
                                self.params.img_width // self.params.patch_size,
                                self.params.img_width // self.params.patch_size,
                                self.params.patch_size ** 2 * self.params.img_channel))
        real_input_flag = torch.FloatTensor(real_input_flag).to(self.device)
        val_pbar = tqdm(val_loader)
        mse_loss, preds, trues = [], [], []
        for i, (batch_x, batch_y,_) in enumerate(val_pbar):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            # eval model
            batch_x = torch.cat([batch_x, batch_y], dim=1)
            pred, loss = self.model(batch_x, real_input_flag.permute(0, 1, 4, 2, 3).contiguous())
            # eval model
            pred = pred[:, :10]
            true, pred_y = batch_y.detach().cpu(), pred.detach().cpu()
            val_pbar.set_description('vali loss: {:.4f}'.format(loss.item()))
            mse_loss.append(loss.item())

            preds.append(pred_y.numpy())
            trues.append(true.numpy())

        mse_loss = np.average(mse_loss)

        preds = np.concatenate(preds,axis=0)
        trues = np.concatenate(trues,axis=0)

        import sys; sys.path.append('/usr/data/gzy/Weather_Forecast')
        from API.metrics import metric
        mae, mse, rmse, mape, mspe,ssim,psnr = metric(preds, trues,val_loader.dataset.mean,val_loader.dataset.std,return_ssim_psnr=True)
        return mse, mae, ssim

    def validate(self, val_loader):
        self.model.eval()
        real_input_flag = np.zeros(
                                (self.params.batch_size,
                                # self.params.total_length - self.params.input_length - 1,
                                self.params.total_length - self.params.input_length,
                                self.params.img_width // self.params.patch_size,
                                self.params.img_width // self.params.patch_size,
                                self.params.patch_size ** 2 * self.params.img_channel))
        real_input_flag = torch.FloatTensor(real_input_flag).to(self.device)
        val_pbar = tqdm(val_loader)
        mse_loss, preds, trues = [], [], []
        number=0
        for i, (batch_x, batch_y,_) in enumerate(val_pbar):
            number+=batch_x.shape[0]
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            # eval model
            batch_x = torch.cat([batch_x, batch_y], dim=1)
            pred, loss = self.model(batch_x, real_input_flag.permute(0, 1, 4, 2, 3).contiguous())
            # eval model
            pred = pred[:, :10]
            true, pred_y = batch_y.detach().cpu(), pred.detach().cpu()
            val_pbar.set_description('vali loss: {:.4f}'.format(loss.item()))
            mse_loss.append(loss.item())

            preds.append(pred_y.numpy())
            trues.append(true.numpy())

            if number>=1000:
                break

        preds = np.concatenate(preds,axis=0)
        trues = np.concatenate(trues,axis=0)

        import sys; sys.path.append('/usr/data/gzy/Weather_Forecast')
        from API.metrics import metric
        mae, mse, rmse, mape, mspe,ssim,psnr = metric(preds, trues,val_loader.dataset.mean,val_loader.dataset.std,return_ssim_psnr=True)
        return mse, mae
        