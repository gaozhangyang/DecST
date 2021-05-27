import torch
import argparse
device = torch.device('cuda:0')

from collections import OrderedDict
from modules.rnn_cell import ConvLSTM_cell

dataset_params = {
    'batch_size': 64,
    'val_batch_size': 64,
    'data_root': '/usr/data/video_dataset/data/'
}

ConvLSTM_params = {
    'encoder_params' : [[
                            OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
                            OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
                            OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
                        ],
                        [
                            ConvLSTM_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=64),
                            ConvLSTM_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=96),
                            ConvLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96)
                        ]],
    'decoder_params' : [[
                            OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
                            OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
                            OrderedDict({
                                'conv3_leaky_1': [64, 16, 3, 1, 1],
                                'conv4_leaky_1': [16, 1, 1, 1, 0]
                            }),
                        ],
                        [
                            ConvLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96),
                            ConvLSTM_cell(shape=(32,32), input_channels=96, filter_size=5, num_features=96),
                            ConvLSTM_cell(shape=(64,64), input_channels=96, filter_size=5, num_features=64),
                        ]],
    'lr': 1e-4,
    'device': device,
}

PredRNN_params = {
    'lr': 1e-3,
    'device': device,
    'num_hidden': [64, 64, 64, 64],
    'patch_size': 4,
    'img_channel': 1,
    'img_width': 64,
    'filter_size': 5,
    'stride': 1,
    'layer_norm': 1,
    'total_length': 20,
    'reverse_scheduled_sampling': 0,
    'input_length': 10,
    'max_iterations': 80000,
    'sampling_start_value': 1.0,
    'sampling_stop_iter': 50000,
    'sampling_changing_rate': 0.00002,
    'scheduled_sampling': 1,
    'batch_size': dataset_params['batch_size']
}

MIM_params = {
            'lr': 0.001,
            'device': torch.device('cuda:0'),
            'num_hidden': [64,64,64,64],
            'patch_size': 1,
            'img_channel': 1,
            'img_width': 64,
            'filter_size': 5,
            'stride': 1,
            'layer_norm': True,
            'total_length': 20,
            'reverse_scheduled_sampling': 0,
            'input_length': 10,
            'max_iterations': 80000,
            'sampling_start_value': 1.0,
            'sampling_stop_iter': 50000,
            'sampling_changing_rate': 0.00002,
            'scheduled_sampling': 1,
            'batch_size': 64
}
if __name__ == '__main__':
    from dataloader import load_data
    train_loader, val_loader, test_loader, loader_mean, loader_std = load_data(dataset_params['batch_size'], \
                                                                               dataset_params['val_batch_size'], 
                                                                               dataset_params['data_root'])

    from modules.mim import MIM
    params = argparse.Namespace(**MIM_params)
    algo = MIM(params)

    for epoch in range(10):
        algo.train(train_loader, epoch)
        algo.evaluate(val_loader)
        algo.validate(val_loader)

    # from modules.rnn_cell import MIMBlock
    # model = MIMBlock('a', 5, 64, 64, [2, 10, 1, 64, 64], 64).cuda()
    # loss_fn = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # for epoch in range(10):
    #     x = torch.rand(2, 64, 64, 64).cuda()
    #     diff_h = torch.rand(2, 64, 64, 64).cuda()
    #     h = torch.rand(2, 64, 64, 64).cuda()
    #     c = torch.rand(2, 64, 64, 64).cuda()
    #     m = torch.rand(2, 64, 64, 64).cuda()
    #     y = torch.rand(2, 64, 64, 64).cuda()
    #     pred = model(x, diff_h, h, c, m)
    #     optimizer.zero_grad()
    #     loss = loss_fn(pred[2], y) # Wrong: [0, 1] ; Correct [2]
    #     loss.backward()
    #     optimizer.step()
    #     print(loss.item())

    # from modules.rnn_cell import MIMN
    # model = MIMN('a', 5, 64, [2, 10, 1, 64, 64]).cuda()
    # loss_fn = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # for epoch in range(10):
    #     x = torch.rand(2, 64, 64, 64).cuda()
    #     h_t = torch.rand(2, 64, 64, 64).cuda()
    #     c_t = torch.rand(2, 64, 64, 64).cuda()
    #     y = torch.rand(2, 64, 64, 64).cuda()
    #     pred = model(x, h_t, c_t)
    #     optimizer.zero_grad()
    #     loss = loss_fn(pred[0], y) # Correct: ALL
    #     loss.backward()
    #     optimizer.step()


    # from modules.rnn_cell import MIM_SpatioTemporalLSTMCell
    # model = MIM_SpatioTemporalLSTMCell('a', 5, 64, 64, [2, 10, 1, 64, 64], 1).cuda()
    # loss_fn = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # for epoch in range(10):
    #     x = torch.rand(2, 1, 64, 64).cuda()
    #     h = torch.rand(2, 64, 64, 64).cuda()
    #     c = torch.rand(2, 64, 64, 64).cuda()
    #     m = torch.rand(2, 64, 64, 64).cuda()
    #     y = torch.rand(2, 64, 64, 64).cuda()
    #     pred = model(x, h, c, m)
    #     optimizer.zero_grad()
    #     loss = loss_fn(pred[1], y) # Correct: ALL
    #     loss.backward()
    #     optimizer.step()
        