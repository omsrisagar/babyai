import os
import torch.nn as nn
import torch.distributed

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "45678"
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
world_size = 1
rank = 0

torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
device = 'cuda'

class BatchNorm_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(64)
        self.rnn = nn.LSTMCell(64, 64)  # input_size, hidden_size

    def forward(self, input, mem_h, mem_c):
        input = self.bn(input)
        mem_h, mem_c = self.rnn(input, (mem_h, mem_c))
        return (mem_h, mem_c)

net = BatchNorm_LSTM()
net.to(device=device)
net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
net = torch.nn.parallel.DistributedDataParallel(
    net,
    device_ids=[rank],
    output_device=rank,
)
net.train()

with torch.autograd.set_detect_anomaly(True):
    input = torch.rand(2, 64, 64, device=device) #time_steps x batch_size x input_size
    mem_h = torch.zeros(64, 64, device=device) # batch_size x hidden_size
    mem_c = torch.zeros(64, 64, device=device) # batch_size x hidden_size
    for i in range(input.size()[0]):
        mem_h, mem_c = net(input[i], mem_h, mem_c)
    out = mem_h.mean()
    out.backward() # throws error!
    print('Passed!') # never reached :(