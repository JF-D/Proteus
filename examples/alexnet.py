import os
import json
import argparse
import proteus
import proteus.torchapi as torch
import proteus.torchapi.nn as nn
from proteus.simulator.simulator import Simulator

# from proteus.torchapi.nn.cube import DeviceCube
# from proteus.ir import ProteusModel, graph
# from proteus.algorithm.dm_algo import DMAlgo
# from proteus.algorithm.ilp_algo import ILPAlgo
# from proteus.algorithm.ilp_stage import StageILP

from models import AlexNet, inception_v3, resnet50, resnet18, vgg19
from build_dev_topo import build_cluster

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='alexnet')
parser.add_argument('-bs', type=int, default=128)
parser.add_argument('-ps', type=str, default='manual')
parser.add_argument('-ndev', type=int, default=4)
parser.add_argument('-cluster', type=str, default='n1_g1')
parser.add_argument('--disable-collective', action='store_true')
parser.add_argument('--bucket-size', type=int, default=25)
parser.add_argument('--reprofile', action='store_true')
parser.add_argument('--profile-iters', type=int, default=10)
parser.add_argument('--test', action='store_true')
parser.add_argument('--flexflow', action='store_true')
args = parser.parse_args()

if __name__ == '__main__':
    # algorithm
    if args.model.lower() == 'alexnet':
        model = AlexNet()
        img_shape = (3, 224, 224)
    elif args.model.lower() == 'inception_v3':
        model = inception_v3(aux_logits=False)
        img_shape = (3, 224, 224)
    elif args.model.lower() == 'resnet50':
        model = resnet50()
        img_shape = (3, 224, 224)
    elif args.model.lower() == 'resnet18':
        model = resnet18()
        img_shape = (3, 224, 224)
    elif args.model.lower() == 'vgg19':
        model = vgg19()
        img_shape = (3, 224, 224)
    else:
        print('Unknown model: {}'.format(args.model))
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # build device topo cluster
    with open(args.cluster, 'r') as f:
        cluster_info = json.load(f)
    cluster = build_cluster(topo_file='{}/topos/topo-n{}.xml'.format(
        os.path.dirname(args.cluster), cluster_info['n_gpu_per_node']),
                            **cluster_info)

    bs = args.bs * cluster.n_node * cluster.n_gpu_per_node
    inputs = {
        'input': (tuple([bs] + list(img_shape)), ),  # tuple of input shape
        'label': ((bs, ), )  # tuple of label shape
    }
    graph, stree = proteus.compile(model, inputs, criterion, optimizer)

    args.ndev = cluster.n_node * cluster.n_gpu_per_node

    stree.init_config(cluster.dev_topo, stride=2)
    dev_topo = stree.dev_topo
    dp_mesh = dev_topo.create_mesh((args.ndev, ))
    if args.ps == 'manual':
        stree.root.split(0, args.ndev)
        stree.root.map(dp_mesh)

        if args.model.lower() == 'alexnet':
            stree.classifier.seq0.split(0, 1, item='out')
            stree.classifier.seq0.map(dp_mesh, item='out')
            stree.classifier.seq0.split(0, args.ndev, item='out_grad')
            stree.classifier.seq0.map(dp_mesh, item='out_grad')
            stree.classifier.seq1.split(1, args.ndev)
            stree.classifier.seq2.split(1, args.ndev)
            stree.classifier.seq3.split(1, args.ndev)
            stree.classifier.seq4.split(2, args.ndev)
            stree.classifier.seq4.split(0, args.ndev, item='out')
            stree.classifier.seq4.map(dp_mesh, item='out')
            stree.classifier.seq5.split(0, args.ndev)
            stree.classifier.seq6.split(0, args.ndev)
        elif args.model.lower() in ['resnet50', 'inception_v3']:
            stree.fc.split(1, args.ndev)
            stree.fc.map(dp_mesh)
            stree.fc.split([0, 1], [1, 1], item='out')
            stree.fc.map(dp_mesh, item='out')
        elif args.model.lower() == 'vgg19':
            stree.seq2.split(0, 1, item='out')
            stree.seq2.map(dp_mesh, item='out')
            stree.seq2.split(0, args.ndev, item='out_grad')
            stree.seq2.map(dp_mesh, item='out_grad')
            stree.classifier.seq0.split(1, args.ndev)
            stree.classifier.seq1.split(1, args.ndev)
            stree.classifier.seq2.split(1, args.ndev)
            stree.classifier.seq3.split(2, args.ndev)
            stree.classifier.seq3.split(0, args.ndev, item='out')
            stree.classifier.seq3.map(dp_mesh, item='out')

            stree.classifier.seq4.split(0, args.ndev)
            stree.classifier.seq5.split(0, args.ndev)
            stree.classifier.seq6.split(0, args.ndev)

            # stree.classifier.seq5.split(0, 1, item='out')
            # stree.classifier.seq5.map(dp_mesh, item='out')
            # stree.classifier.seq5.split(0, args.ndev, item='out_grad')
            # stree.classifier.seq5.map(dp_mesh, item='out_grad')

            # stree.classifier.seq6.split(1, args.ndev)
            # stree.classifier.seq6.map(dp_mesh)
            # stree.classifier.seq6.split([0, 1], [1, 1], item='out')
            # stree.classifier.seq6.map(dp_mesh, item='out')

        stree.schedule()
    elif args.ps == 'dp':
        stree.root.split(0, args.ndev)
        stree.root.map(dp_mesh)

        # offload example
        # cpu_mesh = dev_topo.create_mesh((args.ndev, ), type='cpu')
        # stree.classifier.split(0, args.ndev, item='weight')
        # stree.classifier.map(cpu_mesh, item='weight')

        stree.optimizer.split(0, 1)
        stree.optimizer.map(dp_mesh)
        stree.schedule()
    elif args.ps == 'pp':
        devs = [i for i in range(args.ndev)]
        stg_1 = dev_topo.make_mesh(devs[:args.ndev//2])
        stree.features.split(0, len(stg_1))
        stree.features.map(stg_1)

        stg_2 = dev_topo.make_mesh(devs[args.ndev//2:])
        stree.avgpool.split(0, len(stg_2))
        stree.seq2.split(0, len(stg_2))
        stree.classifier.split(0, len(stg_2))
        stree.criterion.split(0, len(stg_2))
        stree.avgpool.map(stg_2)
        stree.seq2.map(stg_2)
        stree.classifier.map(stg_2)
        stree.criterion.map(stg_2)

        stree.schedule(n_macro_batch=2,
                       interleave_freq=1,
                       max_ongoing_macro_batch=2)
    if args.disable_collective:
        stree.disable_collective_comm()
    stree.set_bucket_size(args.bucket_size)

    graph.init_config(stree)
    stree.propagate(graph)
    graph.propagate({})
    graph.symmetric_forward_backward()

    # graph.to_graphviz()

    graph.export_config('config.txt')

    stree.dump_tree(config=False)

    if 'titan' in args.cluster or '1080' in args.cluster:
        overlap_factor = 0.3
    else:
        overlap_factor = 0.3
    if args.flexflow:
        overlap_factor = 0
    sim = Simulator(graph,
                    stree,
                    cost_type='profile',
                    reprofile=args.reprofile,
                    profile_iters=args.profile_iters,
                    optimizer_overlap=False,
                    cprofile_compile=False,
                    share_bandwidth=(not args.flexflow),
                    overlap_factor=overlap_factor,
                    FlexFlow=args.flexflow)
    cost = sim.run('log/trace', cprofile_analysis=False)

    sim.print_stats()
