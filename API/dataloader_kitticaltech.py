# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""KTH Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import os.path as osp
import random
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import hickle as hkl
from scipy.misc import imresize # requires scipy==1.1.0

# cite the `process_im` code from PredNet, Thanks!
# https://github.com/coxlab/prednet/blob/master/process_kitti.py
def process_im(im, desired_sz): 
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    return im

logger = logging.getLogger(__name__)

def get_binary_mask(x):
    error = torch.abs(x-x[:,0:1]).sum(dim=1)
    binary_mask = error<0.05
    y = binary_mask[0]
    for t in range(1,binary_mask.shape[0]):
        y = y&binary_mask[t]
    return y

class KittiCaltechDataset(Dataset):
    def __init__(self, datas, indices, pre_seq_length, aft_seq_length, require_back=False):
        super(KittiCaltechDataset,self).__init__()
        self.datas = datas.swapaxes(2, 3).swapaxes(1,2)
        self.indices = indices
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.require_back = require_back
        self.mean = 0
        self.std = 1
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        batch_ind = self.indices[i]
        begin = batch_ind
        end1 = begin + self.pre_seq_length
        end2 = begin + self.pre_seq_length + self.aft_seq_length
        data = torch.tensor(self.datas[begin:end1,::]).float()
        labels = torch.tensor(self.datas[end1:end2,::]).float()
        if self.require_back:
            b_mask = get_binary_mask(data)
            H,W = b_mask.shape
            return data, labels, b_mask.view(1, 1, H, W)
        else:
            return data, labels


class DataProcess(object):
  """Class for preprocessing dataset inputs."""

  def __init__(self, input_param):
    self.paths = input_param['paths']
    self.seq_len = input_param['seq_length']

  def load_data(self, mode='train'):
    """Loads the dataset.

    Args:
      paths: paths of train/test dataset.
      mode: Training or testing.

    Returns:
      A dataset and indices of the sequence.
    """
    if mode == 'train' or mode == 'val': # or mode == 'test':   # train on Kitti dataset 
      kitti_root = self.paths['kitti']
      data = hkl.load(osp.join(kitti_root, 'X_' + mode + '.hkl'))
      data = data.astype('float') / 255.0
      fileidx = hkl.load(osp.join(kitti_root, 'sources_' + mode + '.hkl'))

      indices = []
      index = len(fileidx) - 1
      while index >= self.seq_len - 1:
        if fileidx[index] == fileidx[index - self.seq_len + 1]:
          indices.append(index - self.seq_len + 1)
          index -= self.seq_len - 1
        index -= 1

    elif mode == 'test':  # test on Caltech dataset
      caltech_root = self.paths['caltech']
      data = []
      fileidx = []
      for seq_id in os.listdir(caltech_root):
        for item in os.listdir(osp.join(caltech_root, seq_id)):
          cap = cv2.VideoCapture(osp.join(caltech_root, seq_id, item))
          cnt_frames = 0
          while True:
            ret, frame = cap.read()
            if not ret:
              break
            cnt_frames += 1
            if cnt_frames % 3 == 0: # 10 frames per second
              # frame = cv2.resize(frame, (160, 128)) / 255.0
              frame = process_im(frame, (128, 160)) / 255.0
              data.append(frame)
              fileidx.append(seq_id + item)
          # TEST
        #   break
        # break
      data = np.asarray(data)
      

      indices = []
      index = len(fileidx) - 1
      while index >= self.seq_len - 1:
        if fileidx[index] == fileidx[index - self.seq_len + 1]:
          indices.append(index - self.seq_len + 1)
          index -= self.seq_len - 1
        index -= 1
      
    print(mode + ': there are ' + str(data.shape[0]) + ' pictures')
    print(mode + ': there are ' + str(len(indices)) + ' sequences')

    return data, indices

def load_data(batch_size, val_batch_size, data_root, pre_seq_length, aft_seq_length):
    input_param = {
        'paths': {'kitti': '/usr/data/video_dataset/data/kitti_hkl/', 
                  'caltech': '/usr/data/video_dataset/data/caltech/' },
        'seq_length': (pre_seq_length + aft_seq_length),
        'input_data_type': 'float32',
    }
    input_handle = DataProcess(input_param)
    train_data, train_idx = input_handle.load_data('train')
    # val_data, val_idx = input_handle.load_data('val')
    test_data, test_idx = input_handle.load_data('test')


    train_set = KittiCaltechDataset( train_data,
                              train_idx,
                              pre_seq_length,
                              aft_seq_length,
                              require_back=True)
    # val_set = KittiCaltechDataset( val_data,
    #                           val_idx,
    #                           pre_seq_length,
    #                           aft_seq_length,
    #                           require_back=True)
    test_set = KittiCaltechDataset( test_data,
                              test_idx,
                              pre_seq_length,
                              aft_seq_length,
                              require_back=True)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)
    # dataloader_validation = torch.utils.data.DataLoader(
    #     val_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=1)

    return dataloader_train, dataloader_test, dataloader_test, 0, 1


if __name__ == '__main__':
  pass