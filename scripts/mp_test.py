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

def all_gather(q, ws, device):
    """
    Gathers tensor arrays of different lengths across multiple gpus

    Parameters
    ----------
        q : tensor array
        ws : world size
        device : current gpu device

    Returns
    -------
        all_q : list of gathered tensor arrays from all the gpus

    """
    local_size = torch.tensor(q.size(), device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(all_sizes)

    size_diff = max_size.item() - local_size.item()
    if size_diff:
        padding = torch.zeros(size_diff, device=device, dtype=q.dtype)
        q = torch.cat((q, padding))

    all_qs_padded = [torch.zeros_like(q) for _ in range(ws)]
    dist.all_gather(all_qs_padded, q)
    all_qs = []
    for q, size in zip(all_qs_padded, all_sizes):
        all_qs.append(q[:size])
    return all_qs


def train(gpu, args):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.ws,
        rank=gpu
    )
    device = torch.device(gpu)
    print(f'hello! {gpu}')
    if gpu == 0:
        q = torch.tensor([1.5, 2.3], device=device)
    else:
        q = torch.tensor([5.3], device=device)

    # using pad and truncate
    all_q = all_gather(q, args.ws, device)

    # # using direct all_gather
    # all_q = [torch.zeros_like(q) for _ in range(args.ws)]
    # dist.all_gather(all_q, q)
    # # dist.all_gather_multigpu(all_q, q) # gives RuntimeError: Tensor list operands to scatter/gather must have the same length
    print(all_q)

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
