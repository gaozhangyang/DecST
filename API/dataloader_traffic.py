import numpy as np
from torch.utils.data import Dataset
import torch
import warnings
warnings.filterwarnings("ignore")

# https://arxiv.org/pdf/1811.07490.pdf

def get_binary_mask(x):
    error = torch.abs(x-x[:,0:1]).sum(dim=1)
    binary_mask = error<0.05
    y = binary_mask[0]
    for t in range(1,binary_mask.shape[0]):
        y = y&binary_mask[t]
    return y

class TrafficDataset(Dataset):
    def __init__(self, X,Y,mean=None, std=None, require_back=False):
        super(TrafficDataset,self).__init__()
        self.X = (X+1)/2
        self.Y = (Y+1)/2
        self.mean = -1
        self.std = 2
        
        self.require_back=require_back
        # if require_back:
        #     backgrounds = torch.load('/usr/data/gzy/Weather_Forecast/ex_sota/backgrounds/traffic.pt')
        #     backgrounds = backgrounds.swapaxes(2,3).swapaxes(1,2)

        #     N1 = self.X.shape[0]
        #     x = self.X[:,0,::].reshape(N1,-1)
        #     x = x/np.linalg.norm(x,axis=1,keepdims=True)
            
        #     N2 = backgrounds.shape[0]
        #     b = backgrounds.reshape(N2,-1)
        #     b = b/np.linalg.norm(b,axis=1,keepdims=True)
            
        #     S = np.matmul(x,b.T)
        #     self.back_idx = np.argmax(S,axis=1)
        #     self.backgrounds = backgrounds[:,np.newaxis,::]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index,::]).float()
        labels = torch.tensor(self.Y[index,::]).float()
        if self.require_back:
            b_mask = get_binary_mask(data)
            H,W = b_mask.shape
            return data, labels, b_mask.view(1,1,H,W)
        else:
            return data, labels

def load_data(
        batch_size, val_batch_size,
        data_root,require_back=False):
    
    # X_train, Y_train, X_test, Y_test, mmn, timestamp_train, timestamp_test=load_traffic_data(data_root,len_closeness=4,len_pred=4,len_test=500)
    dataset = np.load(data_root+'TaxiBJ/dataset.npz')
    X_train, Y_train, X_test, Y_test = dataset['X_train'], dataset['Y_train'], dataset['X_test'], dataset['Y_test']
    train_set = TrafficDataset(X=X_train,
                              Y=Y_train,
                              require_back=require_back)

    test_set = TrafficDataset( X=X_test,
                              Y=Y_test,
                              mean=train_set.mean,
                              std=train_set.std,
                              require_back=require_back)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=20)

    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=20)

    return dataloader_train, None, dataloader_test, train_set.mean, train_set.std

if __name__ == '__main__':
    dataloader_train, _, _, _, _ = load_data(batch_size=128, val_batch_size=128, data_root='/usr/data/video_dataset/data/')
    for item in dataloader_train:
        print(item[0].shape)
        break