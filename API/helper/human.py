__author__ = 'gaozhangyang'
import numpy as np
import os
import cv2
import logging
import random
from tqdm import tqdm
from torch.utils.data import Dataset
import torch

logger = logging.getLogger(__name__)

class InputHandle:
    def __init__(self, datas, indices, name, minibatch_size,image_width,channel,seq_length):
        self.name = name
        self.input_data_type = 'float32'
        self.minibatch_size = minibatch_size
        self.image_width = image_width
        self.channel = channel
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = seq_length
        self.interval = 2

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        if self.current_position + self.minibatch_size > self.total():
            return True
        else:
            return False

    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_width, self.image_width, self.channel)).astype(
            self.input_data_type)
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length * self.interval
            print(begin,end)
            data_slice = self.datas[begin:end:self.interval]
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
            # logger.info('data_slice shape')
            # logger.info(data_slice.shape)
            # logger.info(input_batch.shape)
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_position))
        logger.info("    Minibatch Size: " + str(self.minibatch_size))
        logger.info("    total Size: " + str(self.total()))
        logger.info("    current_input_length: " + str(self.current_input_length))
        logger.info("    Input Data Type: " + str(self.input_data_type))


class DataProcess:
    def __init__(self, paths, name, minibatch_size,image_width,channel,seq_length,sv_data=None):
        self.paths = paths
        self.name = name
        self.minibatch_size = minibatch_size
        self.image_width = image_width
        self.channel = channel
        self.seq_len = seq_length
        self.sv_data = sv_data
        if self.sv_data is not None:
            dataset = np.load(self.sv_data)
            self.train_data = dataset['train_data']
            self.train_indices = dataset['train_indices']
            self.test_data = dataset['test_data']
            self.test_indices = dataset['test_indices']


    def load_data(self, paths, mode='train'):
        data_dir = paths[0]
        intervel = 2

        frames_np = []
        scenarios = ['Walking']
        if mode == 'train':
            subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        elif mode == 'test':
            subjects = ['S9', 'S11']
        else:
            print ("MODE ERROR")
        _path = data_dir
        print ('load data...', _path)
        filenames = os.listdir(_path)
        filenames.sort()
        print ('data size ', len(filenames))
        frames_file_name = []
        for filename in tqdm(filenames):
            fix = filename.split('.')
            fix = fix[0]
            subject = fix.split('_')
            scenario = subject[1]
            subject = subject[0]
            if subject not in subjects or scenario not in scenarios:
                continue
            file_path = os.path.join(_path, filename)
            image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
            #[1000,1000,3]
            image = image[image.shape[0]//4:-image.shape[0]//4, image.shape[1]//4:-image.shape[1]//4, :]
            if self.image_width != image.shape[0]:
                image = cv2.resize(image, (self.image_width, self.image_width))
            frames_np.append(np.array(image, dtype=np.float32) / 255.0)
            frames_file_name.append(filename)
        # is it a begin index of sequence
        indices = []
        index = 0
        print ('gen index')
        while index + intervel * self.seq_len - 1 < len(frames_file_name):
            # 'S11_Discussion_1.54138969_000471.jpg'
            # ['S11_Discussion_1', '54138969_000471', 'jpg']
            start_infos = frames_file_name[index].split('.')
            end_infos = frames_file_name[index+intervel*(self.seq_len-1)].split('.')
            if start_infos[0] != end_infos[0]:
                index += 1
                continue
            start_video_id, start_frame_id = start_infos[1].split('_')
            end_video_id, end_frame_id = end_infos[1].split('_')
            if start_video_id != end_video_id:
                index += 1
                continue
            if int(end_frame_id) - int(start_frame_id) == 5 * (self.seq_len - 1) * intervel:
                indices.append(index)
            if mode == 'train':
                index += 10
            elif mode == 'test':
                index += 5
        print("there are " + str(len(indices)) + " sequences")
        # data = np.asarray(frames_np)
        data = frames_np
        print("there are " + str(len(data)) + " pictures")
        return data, indices

    def get_train_input_handle(self):
        if self.sv_data is not None:
            train_data, train_indices = self.train_data, self.train_indices
        else:
            train_data, train_indices = self.load_data(self.paths, mode='train')
        return InputHandle(train_data, train_indices, self.name, self.minibatch_size,self.image_width,self.channel,self.seq_len)

    def get_test_input_handle(self):
        if self.sv_data is not None:
            test_data, test_indices = self.test_data, self.test_indices
        else:
            test_data, test_indices = self.load_data(self.paths, mode='test')
        return InputHandle(test_data, test_indices, self.name, self.minibatch_size,self.image_width,self.channel,self.seq_len)


def data_provider(dataset_name, data_paths, batch_size,
                  img_width, seq_length=20, is_training=True,sv_data=None):
    '''Given a dataset name and returns a Dataset.
    Args:
        dataset_name: String, the name of the dataset.
        train_data_paths: List, [train_data_path1, train_data_path2...]
        valid_data_paths: List, [val_data_path1, val_data_path2...]
        batch_size: Int
        img_width: Int
        is_training: Bool
    Returns:
        if is_training:
            Two dataset instances for both training and evaluation.
        else:
            One dataset instance for evaluation.
    Raises:
        ValueError: If `dataset_name` is unknown.
    '''

    if dataset_name == 'human':
        input_handle = DataProcess(data_paths,'human',batch_size,img_width,3,seq_length,sv_data)
        test_input_handle = input_handle.get_test_input_handle()
        test_input_handle.begin(do_shuffle=False)
        if is_training:
            train_input_handle = input_handle.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            return train_input_handle, test_input_handle
        else:
            return test_input_handle



if __name__ == '__main__':
#    train_input_handle, test_input_handle=data_provider('human',['/usr/data/gzy/Weather_Forecast/data/human/dataset'],64,128,8,is_training=True)
#    train_data=np.array(train_input_handle.datas)
#    train_indices=train_input_handle.indices
#    test_data=np.array(test_input_handle.datas)
#    test_indices=test_input_handle.indices
#    np.savez('/usr/data/gzy/Weather_Forecast/data/human/dataset.npz',train_data=train_data,train_indices=train_indices,test_data=test_data,test_indices=test_indices)

#    train_input_handle, test_input_handle=data_provider('human',['/usr/data/gzy/Weather_Forecast/data/human/dataset'],64,128,8,is_training=True,sv_data='/usr/data/gzy/Weather_Forecast/data/human/dataset.npz')
#    batch=test_input_handle.get_batch()
   print()
