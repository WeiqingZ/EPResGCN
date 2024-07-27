import torch
import numpy as np
import random
import argparse
from model.model import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ratio', type=float, default=0.8, help='train_ratio')

    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--root_path', type=str, default='./', help='root path of the dataset')
    parser.add_argument('--file_path', type=str, default='D:\\Grom\\river_data\\data_5_inrcr', help='root path of the data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--results', type=str, default='./results/0', help='test result image')

    # network
    parser.add_argument('--p_i', type=int, default=9, help='points_in')
    parser.add_argument('--p_h', type=int, default=64, help='points_hidden')
    parser.add_argument('--p_e', type=int, default=256, help='condition_out')

    parser.add_argument('--p_h_e', type=int, default=512, help='condition_out')
    parser.add_argument('--p_o', type=int, default=256, help='condition_out')

    parser.add_argument('--num_layers', type=int, default=18, help='flow_out')
    parser.add_argument('--o_edge_dim', type=int, default=5, help='o_features_dim of edges')
    parser.add_argument('--edge_dim', type=int, default=256, help='features_dim of edges')

    parser.add_argument('--v_h1_s', type=int, default=256, help='v_hidden1_size')
    parser.add_argument('--v_h2_s', type=int, default=128, help='v_hidden2_size')
    parser.add_argument('--v_out_s', type=int, default=1, help='v_out_size')
    parser.add_argument('--m', type=int, default=1)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.is_training:
        setting = 'batch{}_lr{}_epoch{}'.format(
            args.batch_size,
            args.learning_rate,
            args.train_epochs,
        )

    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    net = DlCfd(args)
    # net.train(setting)
    net.predict(setting)

