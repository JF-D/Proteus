import os
import time
import argparse
import torch
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument('-b', type=str, default='1')
parser.add_argument('-e', type=str, default='1')
parser.add_argument('-niters', type=int, default=50)
args = parser.parse_args()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def string_to_int(string):
    if string.endswith('K'):
        cnt = int(string[:-1]) * 1024
    elif string.endswith('M'):
        cnt = int(string[:-1]) * 1024 * 1024
    elif string.endswith('G'):
        cnt = int(string[:-1]) * 1024 * 1024 * 1024
    else:
        cnt = int(string)
    return cnt


def int_to_string(nbytes):
    if nbytes < 1000:
        return '{}B'.format(nbytes)
    elif nbytes < 1e6:
        return '{:.2f}K'.format(nbytes / 1e3)
    elif nbytes < 1e9:
        return '{:.2f}M'.format(nbytes / 1e6)
    else:
        return '{:.2f}G'.format(nbytes / 1e9)


def profile(rank, world_size):
    setup(rank, world_size)

    begin = string_to_int(args.b)
    end = string_to_int(args.e)

    # warmup
    tensor = torch.ByteTensor(1).cuda()
    for _ in range(10):
        dist.broadcast(tensor, 0)

    latency = []
    factor = 0
    while True:
        nbytes = 2**factor
        if nbytes > end:
            break
        if nbytes < begin:
            continue
        tensor = torch.ByteTensor(nbytes).cuda()
        torch.cuda.synchronize()
        st = time.perf_counter()
        for _ in range(args.niters):
            dist.broadcast(tensor, 0)
        torch.cuda.synchronize()
        ed = time.perf_counter()

        t = (ed - st) * 1e6 / args.niters
        latency.append(t)
        if rank == 0:
            print('{}{:7}: {:.3f}us'.format(' ' * 4, int_to_string(nbytes), t))
        factor += 1
    if rank == 0:
        print(latency)

if __name__ == '__main__':
    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
    world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
    profile(rank, world_size)
