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
    parser.add_argument('--res_dir',default='./results',type=str)  # user change logging文件的地址，保存训练过程中的信息，如：图像
    parser.add_argument('--ex_name', default='Debug', type=str) # user change  实验名，便于区分不同的实验
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=3, type=int) # user change 指定训练模型的gpu,如：[0],[0,1],[0,1,2]
    parser.add_argument('--seed',default=1,type=int)
    parser.add_argument('--search',default=False)

    # dataset parameters
    parser.add_argument('--batch_size',default=64,type=int,help='Batch size')
    parser.add_argument('--val_batch_size',default=64,type=int,help='Batch size')
    parser.add_argument('--data_root',default='/usr/data/video_dataset/data/')
    parser.add_argument('--dataname',default='mmnist')
    
    # model parameters # TODO
    parser.add_argument('--method',default='gST',choices=['gST'])
    parser.add_argument('--in_shape',default=[10,1,64,64],type=int,nargs='*')
    parser.add_argument('--drop',default=0,type=float)
    parser.add_argument('--hidC',default=16,type=int)
    parser.add_argument('--hidT',default=4,type=int)

    parser.add_argument('--XNet',default=True,type=bool)
    parser.add_argument('--Inception',default=True,type=bool)
    parser.add_argument('--Background',default=True,type=bool)
    parser.add_argument('--Sskip',default=False,type=bool)
    
    # Training parameters
    parser.add_argument('--epoch_s', default=0, type=int, help='start epoch')
    parser.add_argument('--epoch_e', default=200, type=int, help='end epoch') # 指定最多训练多少轮，我只训练了10轮。
    parser.add_argument('--log_step', default=5, type=int)
    
    parser.add_argument('--lr',default=0.01,type=float,help='Learning rate') # 学习率，不同学习率对性能影响很大，默认0.01即可。
    parser.add_argument('--patience', default=800,type=int)
    return parser