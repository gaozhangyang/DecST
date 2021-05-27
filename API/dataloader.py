from .dataloader_traffic import load_data as load_BJ
from .dataloader_human import load_data as load_human
from .dataloader_moving_mnist import load_data as load_mmnist
from .dataloader_kth import load_data as load_kth
from .dataloader_kitticaltech import load_data as load_kitticaltech

def load_data(dataname,batch_size, val_batch_size, data_root, require_back=False, pre_seq_length=None, aft_seq_length=None):
    if dataname == 'traffic':
        return load_BJ(batch_size, val_batch_size, data_root, require_back)
    elif dataname == 'human':
        return load_human(batch_size, val_batch_size, data_root, require_back)
    elif dataname == 'mmnist':
        return load_mmnist(batch_size, val_batch_size, data_root, require_back)
    elif dataname == 'kth':
        return load_kth(batch_size, val_batch_size, data_root, pre_seq_length, aft_seq_length)
    elif dataname == 'kitticaltech':
        return load_kitticaltech(batch_size, val_batch_size, data_root, pre_seq_length, aft_seq_length)