"""
Script to train the agent through reinforcment learning.
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import logging
import csv
import json
import gym
import time
import datetime
import torch
import numpy as np
import subprocess
import sentencepiece as spm
import json
import itertools

import babyai
import babyai.utils as utils
import babyai.rl
from babyai.arguments import ArgumentParser
from babyai.model import ACModel
from babyai.evaluate import batch_evaluate
from babyai.utils.agent import ModelAgent
from babyai.rl.utils.env import KGEnv
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def train(gpu, args):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.ws,
        rank=gpu
    )
    print(f'hello! {gpu}')

if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus you plan to use in this node')
    parser.add_argument('-ws', '--ws', default=1, type=int,
                        help='world_size: total number of GPUs/processes you are running across all nodes')
    # args.world_size = args.gpus * args.nodes
    # os.environ['MASTER_ADDR'] = '130.107.72.224' #cuda0003
    os.environ['MASTER_ADDR'] = '130.107.72.207'  # tulsi
    os.environ['MASTER_PORT'] = '8888'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    args = parser.parse_args()

    mp.spawn(train, nprocs=args.gpus, args=(args,))
