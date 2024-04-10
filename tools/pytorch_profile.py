import os
import re
import argparse
import time
import timeit
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
from models import GPTLayer, GPT, AlexNet, resnet50, resnet18, inception_v3, vgg19
from models import get_alexnet_mp, get_resnet50_mp, get_inception3_mp, get_vgg_mp

from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='alexnet')
parser.add_argument('-bs', type=int, default=32)
parser.add_argument('-ps', type=str, default='dp')
parser.add_argument('-n-ins', type=int, default=2)
# GPT
parser.add_argument('-version', type=str, default=None)
parser.add_argument('-nlayer', type=int, default=12)
parser.add_argument('-seq-length', type=int, default=512)
parser.add_argument('-hidden-size', type=int, default=768)
parser.add_argument('-nheads', type=int, default=12)
parser.add_argument('-vocab-size', type=int, default=40478)  # 50257

parser.add_argument('-bucket-size', type=int, default=25)
parser.add_argument('--benchmark', action='store_false')
parser.add_argument('--timeline', action='store_true')
parser.add_argument('--timeline-name', type=str, default=None)
parser.add_argument('--niters', type=int, default=10)
parser.add_argument('--nvprof', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--hook', action='store_true')
parser.add_argument('-launch', type=str, default=None)
parser.add_argument('-master_addr', type=str, default='localhost')
parser.add_argument('-port', type=str, default='12355')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()


def setup(local_rank, rank, world_size):
    # initialize the process group
    torch.cuda.set_device(local_rank)
    if args.launch == 'deepspeed':
        import deepspeed
        deepspeed.init_distributed()
    else:
        dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


MB = 1024 * 1024


class ModelStats:

    def __init__(self, name):
        self.name = name
        self._input_size = None
        self._model_size = None
        self._memory_load = None
        self._work_memory = None

        self.alloc_at_forward = 0
        assert torch.cuda.memory_allocated() == 0
        assert torch.cuda.max_memory_allocated() <= 1e3

    def record_input_size(self, n_ins):
        self.alloc_at_input = torch.cuda.memory_allocated()
        self._input_size = torch.cuda.memory_allocated() / n_ins / MB

    def record_model_size(self):
        self.alloc_at_model = torch.cuda.memory_allocated()
        self._model_size = (self.alloc_at_model - self.alloc_at_input) / MB

    def record_ddp_size(self):
        self.alloc_at_ddp = torch.cuda.memory_allocated()
        self._ddp_size = (self.alloc_at_ddp - self.alloc_at_model) / MB

    def record_memory_load(self):
        self.alloc_at_forward = max(self.alloc_at_forward,
                                    torch.cuda.memory_allocated())
        if hasattr(self, 'alloc_at_ddp'):
            self._memory_load = (self.alloc_at_forward -
                                 self.alloc_at_ddp) / MB
        else:
            self._memory_load = (self.alloc_at_forward -
                                 self.alloc_at_model) / MB

    def record_max_memory(self):
        self.alloc_max = torch.cuda.max_memory_allocated()
        self._work_memory = (self.alloc_max - self.alloc_at_forward) / MB

    def show_memory_stats(self):
        print(
            'Input: {:.3f}MB, Model: {:.3f}MB, MemoryLoad: {:.3f}MB, WorkMemory: {:.3f}MB'
            .format(self._input_size, self._model_size,
                    self._input_size + self._model_size + self._memory_load,
                    self._work_memory))


def profile(local_rank, rank, world_size):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    # torch.backends.cudnn.benchmark = args.benchmark
    # torch.backends.cudnn.deterministic = True

    stats = ModelStats(args.model)
    if args.model in torchvision.models.__dict__:
        x = []
        for _ in range(args.n_ins):
            x_ = torch.randn((args.bs, 3, 224, 224))
            x_ = x_.cuda()
            x.append(x_)
        if not args.test:
            y = []
            for _ in range(args.n_ins):
                y_ = torch.LongTensor(args.bs).random_(0, 1000)
                y_ = y_.cuda()
                y.append(y_)

        stats.record_input_size(args.n_ins)

        if args.model == 'alexnet':
            if args.ps == 'mp':
                model = get_alexnet_mp(cuda=True)
            else:
                model = AlexNet()
        elif args.model == 'resnet18':
            model = resnet18()
        elif args.model == 'resnet50':
            if args.ps == 'mp':
                model = get_resnet50_mp(cuda=True)
            else:
                model = resnet50()
        elif args.model == 'inception_v3':
            if args.ps == 'mp':
                model = get_inception3_mp(cuda=True, aux_logits=False)
            else:
                model = inception_v3(aux_logits=False)
        elif args.model == 'vgg19':
            if args.ps == 'mp':
                model = get_vgg_mp(cuda=True)
            else:
                model = vgg19()
        else:
            model = torchvision.models.__dict__[args.model]()

        model.cuda()
        stats.record_model_size()
        if args.ps == 'dp' and world_size > 1:
            model = DDP(model,
                        device_ids=[local_rank],
                        bucket_cap_mb=args.bucket_size)
        if args.ps == 'mp' and world_size > 1:
            if args.model in ['alexnet', 'vgg19']:
                model.features = DDP(model.features,
                                     device_ids=[local_rank],
                                     bucket_cap_mb=args.bucket_size)
                model.linear = DDP(model.linear,
                                   device_ids=[local_rank],
                                   bucket_cap_mb=args.bucket_size)
            elif args.model in ['resnet50', 'inception_v3']:
                model.features = DDP(model.features,
                                     device_ids=[local_rank],
                                     bucket_cap_mb=args.bucket_size)
        stats.record_ddp_size()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        def profile_run(i, record=False):
            x_ = x[i % args.n_ins]
            y_ = y[i % args.n_ins]
            out = model(x_)
            if record:
                stats.record_memory_load()
            if not args.test:
                optimizer.zero_grad()
                loss = criterion(out, y_)
                loss.backward()
                optimizer.step()

    elif args.model == 'gpt':
        if args.version == 'gpt-1' or args.version == 'layer':
            num_layers = 1 if args.version == 'layer' else 12
            seq_length = 512
            hidden_size = 768
            nheads = 12
            attention_dropout_prob = output_dropout_prob = 0.1
            vocab_size = 40478
        else:
            num_layers = args.nlayer
            seq_length = args.seq_length
            hidden_size = args.hidden_size
            nheads = args.nheads
            attention_dropout_prob = output_dropout_prob = 0.1
            vocab_size = args.vocab_size
        model = GPT(num_layers, seq_length, hidden_size, nheads,
                    attention_dropout_prob, output_dropout_prob, vocab_size)
        x, position_ids, mask = [], [], []
        for _ in range(args.n_ins):
            x_ = torch.randint(0,
                               vocab_size, (args.bs, seq_length),
                               dtype=torch.int64).cuda()
            pos_id_ = torch.arange(seq_length, dtype=torch.long).cuda()
            mask_ = torch.tril(torch.ones(
                (args.bs, seq_length,
                 seq_length))).view(args.bs, 1, seq_length, seq_length).cuda()
            x.append(x_)
            position_ids.append(pos_id_)
            mask.append(mask_)
        if not args.test:
            y = []
            for _ in range(args.n_ins):
                y_ = torch.LongTensor(args.bs,
                                      seq_length).random_(0, vocab_size)
                y_ = y_.cuda()
                y.append(y_)
        stats.record_input_size(args.n_ins)

        model.cuda()
        stats.record_model_size()
        model = DDP(model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    bucket_cap_mb=args.bucket_size)
        stats.record_ddp_size()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        def profile_run(i, record=False):
            x_, pos_id_ = x[i % args.n_ins], position_ids[i % args.n_ins]
            mask_ = mask[i % args.n_ins]
            y_ = y[i % args.n_ins]
            out = model(x_, pos_id_, mask_)
            if record:
                stats.record_memory_load()
            if not args.test:
                optimizer.zero_grad()
                loss = criterion(out.view(-1, vocab_size), y_.view(-1))
                loss.backward()
                optimizer.step()

    if rank == 0:
        print(model)
    if args.hook:
        visited = {}

        cnt = 0

        def hook(module, ins, outs):
            if module in visited:
                return outs
            visited[module] = True
            in_shape = []
            for x in ins:
                in_shape.append(x.size())
            for w in module.parameters():
                in_shape.append(w.size())
            out_shape = []
            if isinstance(outs, (tuple, list)):
                for out in outs:
                    out_shape.append(out.size())
            else:
                out_shape.append(outs.size())
            nonlocal cnt
            print(cnt, module)
            print(' ' * 4, module.__class__.__name__, in_shape, out_shape)
            print(' ' * 4, 'Memory alloc: ',
                  torch.cuda.memory_allocated() / 1024 / 1024, 'MB')
            cnt += 2
            return outs

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Sequential):
                continue
            if len(list(module.children())) != 0:
                continue
            module.register_forward_hook(hook)

    # warmup
    if not args.nvprof:
        for i in range(5):
            profile_run(i, record=True)

    torch.cuda.synchronize()

    if args.timeline:
        if args.nvprof:
            torch.cuda.cudart().cudaProfilerStart()
            for i in range(args.niters):
                profile_run(i)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        else:
            if int(torch.__version__.split('.')[1]) < 10:
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    for i in range(args.niters):
                        profile_run(i)
            else:
                with torch.profiler.profile(
                        on_trace_ready=torch.profiler.
                        tensorboard_trace_handler('./log/{}/n{}'.format(
                            args.model, world_size)),
                        record_shapes=False,
                        profile_memory=True,
                ) as prof:
                    for i in range(args.niters):
                        profile_run(i)
                        prof.step()

            if rank == 0:
                if int(torch.__version__.split('.')[1]) >= 7:
                    print(prof.key_averages().table(
                        sort_by='self_cuda_time_total'))
                else:
                    print(prof.key_averages().table(sort_by='cuda_time_total'))
                if int(torch.__version__.split('.')[1]) < 10:
                    if args.timeline_name is None:
                        args.timeline_name = f'log/{args.model}'
                    prof.export_chrome_trace('{}.json'.format(
                        args.timeline_name))
    else:
        replicas, per_replica = 50, 1
        times = []
        for i in range(replicas):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for j in range(per_replica):
                profile_run(j)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000 / per_replica)
            print('[{}/{}] speed: {:.3f}ms/iter'.format(i, replicas, times[-1]))
        print('speed: {:.3f}ms/iter'.format(np.mean(times)))

    stats.record_max_memory()
    if not args.timeline:
        stats.show_memory_stats()

    cleanup()


if __name__ == '__main__':
    if args.launch == 'slurm':
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        node_list = str(os.environ['SLURM_NODELIST'])
        node_parts = re.findall('[0-9]+', node_list)

        os.environ[
            'MASTER_ADDR'] = f'{node_parts[1]}.{node_parts[2]}.{node_parts[3]}.{node_parts[4]}'
        os.environ['MASTER_PORT'] = str(args.port)
    elif args.launch == 'mpirun':
        local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
        rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
        world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.port
    elif args.launch == 'deepspeed':
        local_rank = args.local_rank
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = args.local_rank

    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

    setup(local_rank, rank, world_size)

    profile(local_rank, rank, world_size)
