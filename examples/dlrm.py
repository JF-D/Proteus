import os
import json
import argparse
import numpy as np
import proteus
import proteus.torchapi as torch
import proteus.torchapi.nn as nn
from proteus.simulator.simulator import Simulator

from models import DLRM
from build_dev_topo import build_cluster

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default=None)
parser.add_argument('-bs', type=int, default=128)
parser.add_argument('-ps', type=str, default='dp')
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
    # build device topo cluster
    with open(args.cluster, 'r') as f:
        cluster_info = json.load(f)
    cluster = build_cluster(topo_file='{}/topos/topo-n{}.xml'.format(
        os.path.dirname(args.cluster), cluster_info['n_gpu_per_node']),
                            **cluster_info)
    ndev = cluster.n_node * cluster.n_gpu_per_node

    # algorithm
    bot_mlp='128-512-512-512-64'
    top_mlp='1024-1024-1024-1024-1'
    emb_size=64
    nindices=100
    emb="1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000"
    if args.ps == 'mp':
        if ndev == 16:
            emb = '500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000'
        elif ndev == 32:
            emb = '250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000-250000'
    interaction="dot"
    arch_interaction_itself=False

    ln_emb = np.fromstring(emb, dtype=int, sep="-")
    ln_bot = np.fromstring(bot_mlp, dtype=int, sep="-")
    num_fea = ln_emb.size + 1  # num sparse + num dense features

    m_den_out = ln_bot[ln_bot.size - 1]
    if interaction == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif interaction == "cat":
        num_int = num_fea * m_den_out
    else:
        assert False
    arch_mlp_top_adjusted = str(num_int) + "-" + top_mlp
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")
    model = DLRM(m_spa=emb_size, ln_emb=ln_emb, ln_bot=ln_bot, ln_top=ln_top,
                 arch_interaction_op=interaction,
                 sigmoid_bot=-1, sigmoid_top=ln_top.size - 2,
                 num_indices_per_lookup=nindices,
                 divide=(args.ps == 'mp' and ndev > 16))
    model.train()
    criterion = nn.CrossEntropyLoss(loss_type='mse')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    bs = args.bs * cluster.n_node * cluster.n_gpu_per_node
    inputs = {
        'input': tuple([(bs, ln_bot[0])] + [(bs, ) for _ in range(ln_emb.size)]),  # tuple of input shape
        'label': ((bs, 1), )  # tuple of label shape
    }
    graph, stree = proteus.compile(model, inputs, criterion, optimizer)

    args.ndev = cluster.n_node * cluster.n_gpu_per_node

    stree.init_config(cluster.dev_topo, stride=2)
    dev_topo = stree.dev_topo
    dp_mesh = dev_topo.create_mesh((args.ndev, ))
    if args.ps == 'dp':
        stree.root.split(0, args.ndev)
        stree.root.map(dp_mesh)

        stree.optimizer.split(0, 1)
        stree.optimizer.map(dp_mesh)
        stree.schedule()
    elif args.ps == 'mp':
        sz_per_mp = ln_emb.size // ndev
        for i in range(ln_emb.size):
            mp_mesh = dp_mesh[i // sz_per_mp:i // sz_per_mp + 1]
            stree.root.children[str(i)].split(0, 1)
            stree.root.children[str(i)].map(mp_mesh)
            stree.root.children[str(i)].split(0, args.ndev, item='out')
            stree.root.children[str(i)].map(dp_mesh, item='out')
            stree.root.children[str(i)].split(0, 1, item='out_grad')
            stree.root.children[str(i)].map(mp_mesh, item='out_grad')
        stree.root.bot_l.split(0, args.ndev)
        stree.root.bot_l.map(dp_mesh)

        stree.root.split(0, args.ndev)
        stree.root.map(dp_mesh)

        stree.schedule(nstages=(1, dp_mesh))

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
        overlap_factor = 0.1
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
