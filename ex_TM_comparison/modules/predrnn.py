import torch
import torch.nn as nn
from .rnn_cell import PredRNN_cell
from tqdm import tqdm
import numpy as np
from .utils import reshape_patch, reshape_patch_back

# Cite from https://github.com/thuml/predrnn-pytorch, Thanks!
class PredRNN_model(nn.Module):
    def __init__(self, configs):
        super(PredRNN_model, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel 
        self.num_hidden = configs.num_hidden
        self.num_layers = len(self.num_hidden)
        cell_list = []

        width = configs.img_width // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else self.num_hidden[i - 1]
            cell_list.append(
                PredRNN_cell(in_channel, self.num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[self.num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = torch.FloatTensor(mask_true).to(frames.device)
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)

        # for t in range(self.configs.total_length - 1):
        for t in range(self.configs.total_length):
            # reverse schedule sampling
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.input_length:
                    net = frames[:, t]
                else:
                    try:
                        B = frames.shape[0]
                        net = mask_true[:B,t - self.configs.input_length] * frames[:B, t] + (1 - mask_true[:B, t - self.configs.input_length]) * x_gen
                    except:
                        B = frames.shape[0]
                        net = mask_true[:B,t - self.configs.input_length] * frames[:B, t] + (1 - mask_true[:B, t - self.configs.input_length]) * x_gen
                    # net = x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames, frames_tensor)
        return next_frames, loss


# from .basic_algo import Basic_algo
class PredRNN:
    def __init__(self, params):
        config = params.__dict__
        config.update({
            # 'lr': 0.0003,
            'lr': 0.002,
            'device': torch.device('cuda:0'),
            'num_hidden': [128,128,128,128],
            'patch_size': 4,
            'img_channel': 3,
            'img_width': 128,
            'filter_size': 5,
            'stride': 1,
            'layer_norm': 0,
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
        self.model = PredRNN_model(params).to(self.device)
        # Basic_algo.__init__(self, model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=4, verbose=True)
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
            batch_x = reshape_patch(batch_x.permute(0, 1, 3, 4, 2), self.params.patch_size)
            self.eta, real_input_flag = self.schedule_sampling(self.eta, self.iterations, self.params)
            self.optimizer.zero_grad()
            pred, loss = self.model(batch_x, real_input_flag)
            loss.backward()
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
        val_pbar = tqdm(val_loader)
        mse_loss, preds, trues = [], [], []
        for i, (batch_x, batch_y,_) in enumerate(val_pbar):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            # eval model
            batch_x = torch.cat([batch_x, batch_y], dim=1)
            batch_x = reshape_patch(batch_x.permute(0, 1, 3, 4, 2), self.params.patch_size)
            pred, loss = self.model(batch_x, real_input_flag)
            pred = reshape_patch_back(pred, self.params.patch_size)
            # eval model
            pred = pred[:, :10].permute(0, 1, 4, 2, 3)
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
        val_pbar = tqdm(val_loader)
        mse_loss, preds, trues = [], [], []
        number=0
        for i, (batch_x, batch_y,_) in enumerate(val_pbar):
            number+=batch_x.shape[0]
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            # eval model
            batch_x = torch.cat([batch_x, batch_y], dim=1)
            batch_x = reshape_patch(batch_x.permute(0, 1, 3, 4, 2), self.params.patch_size)
            pred, loss = self.model(batch_x, real_input_flag)
            pred = reshape_patch_back(pred, self.params.patch_size)
            # eval model
            pred = pred[:, :10].permute(0, 1, 4, 2, 3)
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
        