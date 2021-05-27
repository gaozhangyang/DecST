import sys; sys.path.append('/usr/data/gzy/Weather_Forecast')
import nni
import numpy as np
from ex_TM_comparison.exp_conv import Exp_Traffic
from ex_TM_comparison.parser import create_parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    tuner_params = nni.get_next_parameter()
    config.update(tuner_params)

    Exp = Exp_Traffic

    exp = Exp(args)
    print('>>>>>>>start training >>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train(args)