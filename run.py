import os, sys
from os.path import exists
from datetime import datetime
import argparse
import torch
import time

from utils.logger import Logger
from model import UNet, UNet_small
from train import trainer
from test import tester

'''========================================= options ========================================='''
parser = argparse.ArgumentParser()

# data
parser.add_argument('--dataset', default='ROADS', type=str)
parser.add_argument('--dataroot', default='./data', type=str)
parser.add_argument('--img_H', default=300, type=int)
parser.add_argument('--img_W', default=300, type=int)

# experiment
parser.add_argument('--seed', default=2024, type=int)
parser.add_argument('--exp', default='name_your_experiment', type=str, help='description of the current experiment')
parser.add_argument('--expmode', default='train', type=str, choices=['train', 'test'])

# model
parser.add_argument('--model', default='UNet', type=str, choices=['UNet', 'UNet_small'])

# checkpoints
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--saved', type=str, default='')

# train 
parser.add_argument('--epoch', default=30, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='finetune to get better performance')
parser.add_argument('--lr_decay_epoch', default=[10], type=list)
parser.add_argument('--lr_decay_rate', default=0.1, type=float)
parser.add_argument('--wd', default=1e-3, type=float)
parser.add_argument('--betas', default=[0.9, 0.999])
parser.add_argument('--eps', default=1e-8, type=float)
parser.add_argument('--save_model_interval', type=int, default=10)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--print_log', type=int, default=10)

# test
parser.add_argument('--test_model', type=str, default='best_dice', choices=['epoc50', 'best_acc', 'best_dice'], help='which checkpoint to use for testing?')

# losses
parser.add_argument('--tloss_w', type=float, default=1e-2, help='weight for the topology loss, see paper for optimal values and details')
parser.add_argument('--precal_PD', type=str, default=True, help='whether pre-calculate the persistent diagram for GT images to save training time')

# device
parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--num_workers', type=int, default=0)


def main():
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    args.exp_output_dir = os.path.join('./exp', args.exp)
    if args.expmode == 'test':
        args.checkpoint = os.path.join('./exp', args.exp, args.test_model+'.pth')
        args.exp_output_dir = os.path.join('./exp', args.exp, args.test_model)

    if not exists(args.exp_output_dir):
        os.makedirs(args.exp_output_dir)

    if args.expmode == 'train':
        sys.stdout = Logger(os.path.join(args.exp_output_dir, 'log_{}.txt'.format(datetime.now())), sys.stdout)
    elif args.expmode == 'test':
        sys.stdout = Logger(os.path.join(args.exp_output_dir, 'testlog_{}.txt'.format(datetime.now())), sys.stdout)
    print(args)

    if torch.cuda.is_available():
        args.device = 'cuda:' + args.gpu_id
    else:
        args.device = 'cpu'
    print('Using device:', args.device)

    if args.dataset.startswith('CRACKTREE'):
        args.img_H = 300
        args.img_W = 300
    elif args.dataset.startswith('DRIVE'):
        args.img_H = 275
        args.img_W = 275
    elif args.dataset == 'ROADS':
        args.img_H = 300
        args.img_W = 300
    elif args.dataset == 'ROADS48' or args.dataset == 'Elegan48':
        args.img_H = 48
        args.img_W = 48
    elif args.dataset == 'CREMI':
        args.img_H = 250
        args.img_W = 250

    # model
    if args.model == 'UNet':
        model = UNet(3, 1, Hb=args.img_H // 2 ** 4, Wb=args.img_W // 2 ** 4)
    elif args.model == 'UNet_small':
        model = UNet_small(3, 1, Hb=args.img_H // 2 ** 4, Wb=args.img_W // 2 ** 4)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model = model.to(args.device)

    start_time = time.time()

    if args.expmode == 'train':
        trainer(args, model)
    elif args.expmode == 'test':
        tester(args, model)

    end_time = time.time()
    running_time = end_time - start_time
    running_time_hours = running_time / 3600
    gpu_name = torch.cuda.get_device_name(0)

    if args.expmode == 'train':
        print(f"Ran for {running_time:.2f} seconds ({running_time_hours:.2f} hours) using {gpu_name} on {args.dataset} dataset trained for {args.epoch} epochs.")
    elif args.expmode == 'test' or args.expmode == 'test_whole':
        print(f"Ran for {running_time:.2f} seconds ({running_time_hours:.2f} hours) using {gpu_name} on {args.dataset} dataset.")

if __name__ == '__main__':
    main()






