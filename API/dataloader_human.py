import numpy as np
from torch.utils.data import Dataset
import torch
from .helper.human import DataProcess

def get_binary_mask(x):
    error = torch.abs(x-x[:,0:1]).sum(dim=1)
    binary_mask = error<0.05
    y = binary_mask[0]
    for t in range(1,binary_mask.shape[0]):
        y = y&binary_mask[t]
    return y

class HumanDataset(Dataset):
    def __init__(self, datas, indices,seq_length, require_back=False):
        super(HumanDataset,self).__init__()
        self.datas = datas.swapaxes(2,3).swapaxes(1,2)
        self.indices=indices
        self.current_input_length = seq_length
        self.interval = 2
        self.mean = 0
        self.std = 1

        self.require_back=require_back

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, i):
        batch_ind = self.indices[i]
        begin = batch_ind
        end1 = begin + self.current_input_length//2 * self.interval
        end2 = begin + self.current_input_length * self.interval
        data = torch.tensor(self.datas[begin:end1:self.interval,::]).float()
        labels = torch.tensor(self.datas[end1:end2:self.interval,::]).float()
        if self.require_back:
            b_mask = get_binary_mask(data)
            H,W = b_mask.shape
            return data, labels, b_mask.view(1,1,H,W)
        else:
            return data, labels

def load_data(
        batch_size, val_batch_size,
        data_root,require_back=False):

    input_handle = DataProcess([data_root],'human',64,128,3,8,sv_data=data_root+'human/dataset.npz')
    test_input_handle = input_handle.get_test_input_handle()
    train_input_handle = input_handle.get_train_input_handle()

    train_set = HumanDataset( train_input_handle.datas,
                              train_input_handle.indices,
                              8,require_back)

    test_set = HumanDataset( test_input_handle.datas,
                              test_input_handle.indices,
                              8,require_back)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=20)
    dataloader_validation = None
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=20)

    return dataloader_train, dataloader_validation, dataloader_test, 0, 1


if __name__ == '__main__':
    # input_handle = DataProcess(['/usr/data/gzy/Weather_Forecast/data/human/dataset'],'human',64,128,3,8,sv_data='/usr/data/gzy/Weather_Forecast/data/human/dataset.npz')
    # test_input_handle = input_handle.get_test_input_handle()
    # train_input_handle = input_handle.get_train_input_handle()
    # dataset=HumanDataset(train_input_handle.datas,train_input_handle.indices,8)
    # data = dataset[0]
    # print()

    dataloader_train, dataloader_validation, dataloader_test,_,_ = load_data(64,64,'/usr/data/video_dataset/data/human/dataset')
    print()