import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp

def test(indx):
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            world_size=3,
                            rank=indx)
    # tensor_list = []
    # for dev_idx in range(torch.cuda.device_count()):
    # for dev_idx in range(1):
    # tensor_list.append(torch.FloatTensor([indx]).cuda(indx))
    # tensor_list.append(torch.tensor(indx, device=torch.device(indx)))
    mytensor = torch.tensor([indx*1.0, indx+1.0], device=torch.device(indx)) # device index = 0 on cuda0003

    output_list = [torch.tensor([0.0, 0.0], device=torch.device(indx)) for _ in range(3)]

    # dist.all_gather_multigpu(output_list, tensor_list)
    dist.all_gather(output_list, mytensor)

    print('done')

if __name__ == '__main__':
    # os.environ['MASTER_ADDR'] = '130.107.72.224' #cuda0003
    os.environ['MASTER_ADDR'] = '130.107.72.207'  # tulsi
    # os.environ['MASTER_ADDR'] = '130.107.72.220'  # cuda0001
    os.environ['MASTER_PORT'] = '8888'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    mp.spawn(test, nprocs=2)
    # test(2) # on cuda0003
