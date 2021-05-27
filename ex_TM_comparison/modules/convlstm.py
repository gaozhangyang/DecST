# Cite from https://github.com/jhhuang96/ConvLSTM-PyTorch, Thanks! 
from torch import nn
import torch
from collections import OrderedDict

from .rnn_cell import ConvLSTM_cell
from .rnn_encoder import ConvLSTM_Encoder
from .rnn_decoder import ConvLSTM_Decoder
class ConvLSTM_model(nn.Module):
    def __init__(self, encoder_params, decoder_params):
        super(ConvLSTM_model, self).__init__()
        self.encoder = ConvLSTM_Encoder(encoder_params[0], encoder_params[1])
        self.decoder = ConvLSTM_Decoder(decoder_params[0], decoder_params[1])

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


from .basic_algo import Basic_algo
class ConvLSTM(Basic_algo):
    def __init__(self, params):
        config = params.__dict__
        config.update({
            'encoder_params' : [[
                                    OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
                                    OrderedDict({'conv2_leaky_1': [32, 32, 3, 2, 1]}),
                                    OrderedDict({'conv3_leaky_1': [64, 64, 3, 2, 1]}),
                                ],
                                [
                                    ConvLSTM_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=32),
                                    ConvLSTM_cell(shape=(32,32), input_channels=32, filter_size=5, num_features=64),
                                    ConvLSTM_cell(shape=(16,16), input_channels=64, filter_size=5, num_features=64)
                                ]],
            'decoder_params' : [[
                                    OrderedDict({'deconv1_leaky_1': [64, 64, 4, 2, 1]}),
                                    OrderedDict({'deconv2_leaky_1': [64, 64, 4, 2, 1]}),
                                    OrderedDict({
                                        'conv3_leaky_1': [32, 16, 3, 1, 1],
                                        'conv4_leaky_1': [16, 1, 1, 1, 0]
                                    }),
                                ],
                                [
                                    ConvLSTM_cell(shape=(16,16), input_channels=64, filter_size=5, num_features=64),
                                    ConvLSTM_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=64),
                                    ConvLSTM_cell(shape=(64,64), input_channels=64, filter_size=5, num_features=32),
                                ]],
            'lr': 1e-4,
            'device': torch.device('cuda:0'),
        })
        self.device = params.device
        model = ConvLSTM_model(params.encoder_params, params.decoder_params).to(self.device)
        Basic_algo.__init__(self, model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)
        self.criterion = torch.nn.MSELoss()

    def _iter_batch(self, x, y):
        pred = self.model(x)
        loss = self.criterion(pred, y)
        return pred, loss
        