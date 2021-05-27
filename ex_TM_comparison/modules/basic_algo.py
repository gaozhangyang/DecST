import torch
from torch.nn import MSELoss
import numpy as np
from tqdm import tqdm


class Basic_algo(object):
    def __init__(self, model):
        super(Basic_algo, self).__init__()
        self.model = model
        self.criterion = None

    def _iter_batch(self, x, y):
        pass
        # pred = self.model(x)
        # loss = self.criterion(pred, y)
        # return pred, loss

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
            # train model
            self.optimizer.zero_grad()
            pred_y, loss = self._iter_batch(batch_x, batch_y)
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

    def measure(self, pred, true, loader_mean, loader_std, require_ssim_psnr=True):
        import sys; sys.path.append('/usr/data/gzy/Weather_Forecast')
        from API.metrics import metric
        mae, mse, _, _, _, ssim, psnr = metric(pred, true, \
                                    loader_mean, loader_std, return_ssim_psnr=require_ssim_psnr)
        return mae, mse, ssim, psnr