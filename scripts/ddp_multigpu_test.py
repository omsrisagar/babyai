import torch
import os
import torch.distributed as dist

# os.environ['MASTER_ADDR'] = '130.107.72.224' #cuda0003
os.environ['MASTER_ADDR'] = '130.107.72.207'  # tulsi
# os.environ['MASTER_ADDR'] = '130.107.72.220'  # cuda0001
os.environ['MASTER_PORT'] = '8888'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
dist.init_process_group(backend="nccl",
                        init_method="env://",
                        world_size=3,
                        rank=0)
tensor_list = []
for dev_idx in range(torch.cuda.device_count()):
    tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))

dist.all_reduce_multigpu(tensor_list)

print('done')