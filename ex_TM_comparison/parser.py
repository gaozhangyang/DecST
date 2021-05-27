import argparse


def create_parser():
    """Creates a parser with all the variables that can be edited by the user.

    Returns:
        parser: a parser for the command line
    """
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device',default='cuda',type=str,help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step',default=10,type=int,help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir',default='/usr/data/gzy/Weather_Forecast/ex_TM_comparison/results',type=str)  # user change logging文件的地址，保存训练过程中的信息，如：图像
    parser.add_argument('--ex_name', default='Debug', type=str) # user change  实验名，便于区分不同的实验
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int) # user change 指定训练模型的gpu,如：[0],[0,1],[0,1,2]
    parser.add_argument('--seed',default=1,type=int)
    parser.add_argument('--search',default=False)

    # dataset parameters
    parser.add_argument('--batch_size',default=16,type=int,help='Batch size')
    # parser.add_argument('--batch_size',default=4,type=int,help='Batch size')
    parser.add_argument('--val_batch_size',default=16,type=int,help='Batch size')
    # parser.add_argument('--val_batch_size',default=4,type=int,help='Batch size')
    parser.add_argument('--data_root',default='/usr/data/video_dataset/data/')
    parser.add_argument('--dataname',default='human')
    
    # model parameters # TODO
    parser.add_argument('--model',default='MIM',choices=['ConvLSTM', 'DecST','PhyDNet','PredRNN', 'E3DLSTM', 'MIM','CrevNet'])
    
    # Training parameters
    parser.add_argument('--epoch_s', default=0, type=int, help='start epoch')
    parser.add_argument('--epoch_e', default=1, type=int, help='end epoch') # 指定最多训练多少轮，我只训练了10轮。
    return parser