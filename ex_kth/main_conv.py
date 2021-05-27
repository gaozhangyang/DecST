import nni
import numpy as np
from exp_conv import Exp_KTH
from parser import create_parser
import torch

torch.set_num_threads(4)

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    tuner_params = nni.get_next_parameter()
    config.update(tuner_params)

    Exp = Exp_KTH

    exp = Exp(args)
    print('>>>>>>>start training >>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train(args)
    
    print('>>>>>>>testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse=exp.test(args)

    nni.report_final_result(mse)