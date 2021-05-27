# Cite from https://github.com/metrofun/E3D-LSTM
from functools import reduce
import copy
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rnn_cell import E3DLSTMCell, ConvDeconv3d
from .utils import window
from tqdm import tqdm
import numpy as np


class E3DLSTM_Module(nn.Module):
    def __init__(self, input_shape, hidden_size, num_layers, kernel_size, tau):
        super().__init__()

        self._tau = tau
        self._cells = []

        input_shape = list(input_shape)
        for i in range(num_layers):
            cell = E3DLSTMCell(input_shape, hidden_size, kernel_size)
            # NOTE hidden state becomes input to the next cell
            input_shape[0] = hidden_size
            self._cells.append(cell)
            # Hook to register submodule
            setattr(self, "cell{}".format(i), cell)

    def forward(self, input):
        # NOTE (seq_len, batch, input_shape)
        batch_size = input.size(1)
        c_history_states = []
        h_states = []
        outputs = []

        for step, x in enumerate(input):
            for cell_idx, cell in enumerate(self._cells):
                if step == 0:
                    c_history, m, h = self._cells[cell_idx].init_hidden(
                        batch_size, self._tau, input.device
                    )
                    c_history_states.append(c_history)
                    h_states.append(h)

                # NOTE c_history and h are coming from the previous time stamp, but we iterate over cells
                c_history, m, h = cell(
                    x, c_history_states[cell_idx], m, h_states[cell_idx]
                )
                c_history_states[cell_idx] = c_history
                h_states[cell_idx] = h
                # NOTE hidden state of previous LSTM is passed as input to the next one
                x = h

            outputs.append(h)

        # NOTE Concat along the channels
        return torch.cat(outputs, dim=1)


class E3DLSTM_Model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.encoder = E3DLSTM_Module(params.input_shape, params.hidden_size, params.lstm_layers, params.kernel, params.tau)
        # self.decoder = nn.Conv3d(params.hidden_size * params.time_steps, params.output_shape[0], params.kernel, padding=(0, 2, 2))
        self.decoder = nn.Conv3d(64, 1, kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2))
    def forward(self, x):
        return self.decoder(self.encoder(x))

from .basic_algo import Basic_algo
class E3DLSTM(Basic_algo):
    def __init__(self, params):
        config = params.__dict__
        config.update({
            'input_shape': (3, 4, 128, 128),
            'output_shape': (3, 4, 128, 128),
            'hidden_size': 64,
            'lstm_layers': 4,
            'kernel': (2, 5, 5),
            'tau': 2,
            'temporal_frames': 4,
            'temporal_stride': 1, 
            'input_time_window': 4,
            'output_time_horizon': 1,
            'time_steps': 1, 
            'lr': 0.001,
            'device': torch.device('cuda:0')
        })
        model = E3DLSTM_Model(params).to(params.device)
        Basic_algo.__init__(self, model)
        self.device = params.device
        self.params = params
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)
        self.criterion = torch.nn.MSELoss()

    def _iter_batch(self, x, y):
        pred = self.model(x)
        loss = self.criterion(pred, y)
        return pred, loss

    def train(self, train_loader, epoch): 
        '''
        Train the model with train_loader.
        Input params:
            train_loader: dataloader of train.
        Output params:
            mse_loss: mean square loss between predictions and ground truth.
        '''
        self.model.train()
        train_pbar = tqdm(train_loader)
        mse_loss = []
        for i, (batch_x, batch_y, _) in enumerate(train_pbar):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            batch_x = batch_x.permute(0, 2, 1, 3, 4)
            batch_y = batch_y.permute(0, 2, 1, 3, 4)
            # train model
            self.optimizer.zero_grad()
            frames_seq = []
            for indices in window(range(self.params.input_time_window), \
                    self.params.temporal_frames, self.params.temporal_stride):
                frames_seq.append(batch_x[:, :, indices[0] : indices[-1] + 1])
            batch_x = torch.stack(frames_seq, dim=0)
            pred, loss = self._iter_batch(batch_x, batch_y)
            loss.backward()
            self.optimizer.step()
            # trian model
            train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
            mse_loss.append(loss.item())
        mse_loss = np.average(mse_loss)
        return mse_loss

    def evaluate(self, val_loader):
        '''
        Evaluate the model with val_loader.
        Input params:
            val_loader: dataloader of validation.
        Output params:
            (mse, mae, ssim): mse, mas, ssim between predictions and ground truth.
        '''
        self.model.eval()
        val_pbar = tqdm(val_loader)
        mse_loss, preds, trues = [], [], []
        for i, (batch_x, batch_y, _) in enumerate(val_pbar):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            # eval model
            batch_x = batch_x.permute(0, 2, 1, 3, 4)
            batch_y = batch_y.permute(0, 2, 1, 3, 4)
            frames_seq = []
            for indices in window(range(self.params.input_time_window), \
                    self.params.temporal_frames, self.params.temporal_stride):
                frames_seq.append(batch_x[:, :, indices[0] : indices[-1] + 1])
            batch_x = torch.stack(frames_seq, dim=0)
            pred_y, loss = self._iter_batch(batch_x, batch_y)
            # eval model
            true, pred_y = batch_y.detach().cpu(), pred_y.detach().cpu()
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
        number = 0
        val_pbar = tqdm(val_loader)
        mse_loss, preds, trues = [], [], []
        for i, (batch_x, batch_y, _) in enumerate(val_pbar):
            number += batch_x.shape[0]
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            # eval model
            batch_x = batch_x.permute(0, 2, 1, 3, 4)
            batch_y = batch_y.permute(0, 2, 1, 3, 4)
            frames_seq = []
            for indices in window(range(self.params.input_time_window), \
                    self.params.temporal_frames, self.params.temporal_stride):
                frames_seq.append(batch_x[:, :, indices[0] : indices[-1] + 1])
            batch_x = torch.stack(frames_seq, dim=0)
            pred_y, loss = self._iter_batch(batch_x, batch_y)
            # eval model
            true, pred_y = batch_y.detach().cpu(), pred_y.detach().cpu()
            val_pbar.set_description('vali loss: {:.4f}'.format(loss.item()))
            mse_loss.append(loss.item())

            preds.append(pred_y.numpy())
            trues.append(true.numpy())
            if number >= 1000:
                break

        mse_loss = np.average(mse_loss)

        preds = np.concatenate(preds,axis=0)
        trues = np.concatenate(trues,axis=0)
        import sys; sys.path.append('/usr/data/gzy/Weather_Forecast')
        from API.metrics import metric
        mae, mse, rmse, mape, mspe,ssim,psnr = metric(preds, trues,val_loader.dataset.mean,val_loader.dataset.std,return_ssim_psnr=True)
        return mse, mae