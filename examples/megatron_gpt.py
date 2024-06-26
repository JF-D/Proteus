import os
import argparse
import json
import numpy as np
import proteus
import proteus.torchapi.nn as nn
from proteus.simulator.simulator import Simulator
# from proteus.algorithm.dm_algo import DMAlgo
# from proteus.algorithm.ilp_algo import ILPAlgo
# from proteus.algorithm.ilp_stage import StageILP
from models.transformer import *
from build_dev_topo import build_cluster

parser = argparse.ArgumentParser()
parser.add_argument('-bs', type=int, default=32)
parser.add_argument('-global-bs', type=int, default=None)
parser.add_argument('-n-macro-batch', type=int, default=1)

parser.add_argument('-model', type=str, default='gpt')
parser.add_argument('--no-seq-first', action='store_false', dest='seq_first')
parser.add_argument('-version', type=str, default=None)
parser.add_argument('-nlayer', type=int, default=12)
parser.add_argument('-seq-length', type=int, default=512)
parser.add_argument('-hidden-size', type=int, default=768)
parser.add_argument('-nheads', type=int, default=12)
parser.add_argument('-vocab-size', type=int, default=40478)  # 50257
parser.add_argument('--make-vocab-size-divisible-by', type=int, default=128)
parser.add_argument('-cluster', type=str, default='n1_g1')

parser.add_argument('-ps', type=str, default='dp')
parser.add_argument('-zero', type=int, default=0)
parser.add_argument('-mp-deg', type=int, default=1)
parser.add_argument('-pp-deg', type=int, default=1)
parser.add_argument('--checkpoint', action='store_true')
parser.add_argument('-dom', type=str, default='DM')
parser.add_argument('--disable-collective', action='store_true')
parser.add_argument('--bucket-size', type=int, default=25)
parser.add_argument('--no-share-bandwidth', action='store_true')
parser.add_argument('--reprofile', action='store_true')
parser.add_argument('--profile-iters', type=int, default=10)
parser.add_argument('--test', action='store_true')
parser.add_argument('--flexflow', action='store_true')
args = parser.parse_args()

if __name__ == '__main__':
    # algorithm
    if args.model == 'gpt-1' or args.version == 'layer':
        num_layers = 1 if args.version == 'layer' else 12
        seq_length = 512
        hidden_size = 768
        nheads = 12
        attention_dropout_prob = output_dropout_prob = 0.1
        vocab_size = 40478
    elif args.model == 'gpt-2':
        num_layers = 12
        seq_length = 1024
        hidden_size = 768
        nheads = 12
        attention_dropout_prob = output_dropout_prob = 0.1
        vocab_size = 40478
    elif args.model == 'gpt-1.5b':
        num_layers = 48
        seq_length = 1024
        hidden_size = 1600
        nheads = 16
        attention_dropout_prob = output_dropout_prob = 0.1
        vocab_size = 50257
    else:
        num_layers = args.nlayer
        seq_length = args.seq_length
        hidden_size = args.hidden_size
        nheads = args.nheads
        attention_dropout_prob = output_dropout_prob = 0.1
        vocab_size = args.vocab_size

    # build device topo cluster
    with open(args.cluster, 'r') as f:
        cluster_info = json.load(f)
    cluster = build_cluster(topo_file='{}/topos/topo-n{}.xml'.format(
        os.path.dirname(args.cluster), cluster_info['n_gpu_per_node']),
                            **cluster_info)

    ndev = cluster.n_node * cluster.n_gpu_per_node

    if args.ps == 'megatron_node':
        args.mp_deg = min(ndev, cluster.n_gpu_per_node)
        if args.model == 'gpt-2':
            args.mp_deg = min(args.mp_deg, 4)
        args.ps = 'megatron'

    after = vocab_size
    multiple = args.make_vocab_size_divisible_by * args.mp_deg
    while (after % multiple) != 0:
        after += 1
    args.vocab_size = vocab_size = after

    model = GPT(num_layers, seq_length, hidden_size, nheads,
                attention_dropout_prob, output_dropout_prob, vocab_size, seq_first=args.seq_first)
    model.train(not args.test)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    if args.global_bs is None:
        bs = args.bs * ndev
    else:
        bs = args.global_bs
    inputs = {
        # tuple of input shape
        'input': (tuple([bs, seq_length]), (
            bs,
            seq_length,
        ), tuple([bs, nheads, seq_length, seq_length])),
        'label': ((bs, seq_length), )  # tuple of label shape
    }
    graph, stree = proteus.compile(model, inputs, criterion, optimizer)

    stree.init_config(cluster.dev_topo, stride=2)
    dev_topo = stree.dev_topo

    if args.ps == 'dp':
        assert not args.seq_first
        dp_mesh = dev_topo.create_mesh((ndev, ))
        stree.root.split(0, ndev)
        stree.root.map(dp_mesh)
        stree.optimizer.map(dp_mesh)
        stree.schedule()

    mp_deg = args.mp_deg
    pp_deg = args.pp_deg
    dp_deg = ndev // (mp_deg * pp_deg)
    assert ndev % (mp_deg * pp_deg) == 0
    n_mp_group = ndev // mp_deg
    n_pp_group = ndev // pp_deg
    n_dp_group = ndev // dp_deg

    dp_groups, mp_groups, pp_groups = [], [], []
    for i in range(pp_deg):
        start_rank, end_rank = i * n_pp_group, (i + 1) * n_pp_group
        for j in range(mp_deg):
            ranks = range(start_rank + j, end_rank, mp_deg)
            dp_groups.append(list(ranks))
    for i in range(n_mp_group):
        ranks = list(range(i * mp_deg, (i + 1) * mp_deg))
        mp_groups.append(ranks)

    for i in range(n_pp_group):
        ranks = list(range(i, ndev, n_pp_group))
        pp_groups.append(ranks)
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

    mesh = dev_topo.create_mesh((pp_deg, dp_deg, mp_deg), type='gpu')

    def split_transformer_layer(layer, dmp_mesh):
        dp_deg, mp_deg = dmp_mesh.shape
        dp_mesh = dmp_mesh.reshape(-1)

        if args.seq_first:
            layer.input_layernorm.split(1, dp_deg)
            layer.input_layernorm.map(dmp_mesh)

            layer.attention.qkv.split([1, 2], [dp_deg, mp_deg])
            layer.attention.qkv.map(dmp_mesh)
            layer.attention.seq2.split([1, 2], [dp_deg, mp_deg])
            layer.attention.seq2.map(dmp_mesh)
            layer.attention.seq8.split([0, 1], [dp_deg, mp_deg])
            layer.attention.seq8.map(dmp_mesh)
            # layer.attention.scaled_attention_mask.split(0, dp_mesh.size)
            # layer.attention.scaled_attention_mask.map(dp_mesh)
            layer.attention.scaled_attention_mask.split([0, 1], [dp_deg, mp_deg])
            layer.attention.scaled_attention_mask.map(dmp_mesh)
            layer.attention.seq13.split([0, 1], [dp_deg, mp_deg])
            layer.attention.seq13.map(dmp_mesh)
            layer.attention.seq16.split([0, 1], [dp_deg, mp_deg])
            layer.attention.seq16.map(dmp_mesh)
            layer.attention.dense.split([1, 3], [dp_deg, mp_deg])
            layer.attention.dense.map(dmp_mesh)

            layer.mlp.dense_h_to_4h.split([1, 2], [dp_deg, mp_deg])
            layer.mlp.dense_h_to_4h.map(dmp_mesh)
            layer.mlp.activation.split([1, 2], [dp_deg, mp_deg])
            layer.mlp.activation.map(dmp_mesh)
            layer.mlp.dense_4h_to_h.split([1, 3], [dp_deg, mp_deg])
            layer.mlp.dense_4h_to_h.map(dmp_mesh)

            layer.post_attention_layernorm.split(1, dp_deg)
            layer.post_attention_layernorm.map(dmp_mesh)
            if args.zero >= 2:
                items = ['weight_grad', 'weight'] if args.zero >= 3 else ['weight_grad']
                for key in items:
                    layer.input_layernorm.split(0, dp_deg, item=key)
                    layer.input_layernorm.map(dmp_mesh, item=key)

                    layer.attention.qkv.split(0, dp_mesh.size, item=key)
                    layer.attention.qkv.map(dmp_mesh.transpose().reshape(-1), item=key)
                    layer.attention.dense.split([0, 1], [dp_deg, mp_deg], item=key)
                    layer.attention.dense.map(dmp_mesh, item=key)

                    layer.mlp.dense_h_to_4h.split(0, dp_mesh.size, item=key)
                    layer.mlp.dense_h_to_4h.map(dmp_mesh.transpose().reshape(-1), item=key)
                    layer.mlp.dense_4h_to_h.split([0, 1], [dp_deg, mp_deg], item=key)
                    layer.mlp.dense_4h_to_h.map(dmp_mesh, item=key)

                    layer.post_attention_layernorm.split(0, dp_deg, item=key)
                    layer.post_attention_layernorm.map(dmp_mesh, item=key)

                items = ['bias_grad', 'bias'] if args.zero >= 3 else ['bias_grad']
                for key in items:
                    layer.input_layernorm.split(0, dp_deg, item=key)
                    layer.input_layernorm.map(dmp_mesh, item=key)

                    layer.attention.qkv.split(0, dp_mesh.size, item=key)
                    layer.attention.qkv.map(dmp_mesh.transpose().reshape(-1), item=key)
                    # layer.attention.dense.split(0, dp_deg, item=key)
                    # layer.attention.dense.map(dmp_mesh, item=key)

                    layer.mlp.dense_h_to_4h.split(0, dp_mesh.size, item=key)
                    layer.mlp.dense_h_to_4h.map(dmp_mesh.transpose().reshape(-1), item=key)
                    # layer.mlp.dense_4h_to_h.split(0, dp_deg, item=key)
                    # layer.mlp.dense_4h_to_h.map(dmp_mesh, item=key)

                    layer.post_attention_layernorm.split(0, dp_deg, item=key)
                    layer.post_attention_layernorm.map(dmp_mesh, item=key)
        else:
            layer.input_layernorm.split(0, dp_deg)
            layer.input_layernorm.map(dmp_mesh)

            layer.attention.qkv.split([0, 2], [dp_deg, mp_deg])
            layer.attention.qkv.map(dmp_mesh)
            layer.attention.seq2.split([0, 2], [dp_deg, mp_deg])
            layer.attention.seq2.map(dmp_mesh)
            layer.attention.seq11.split([0, 2], [dp_deg, mp_deg])
            layer.attention.seq11.map(dmp_mesh)
            layer.attention.dense.split([0, 3], [dp_deg, mp_deg])
            layer.attention.dense.map(dmp_mesh)

            layer.mlp.dense_h_to_4h.split([0, 2], [dp_deg, mp_deg])
            layer.mlp.dense_h_to_4h.map(dmp_mesh)
            layer.mlp.activation.split([0, 2], [dp_deg, mp_deg])
            layer.mlp.activation.map(dmp_mesh)
            layer.mlp.dense_4h_to_h.split([0, 3], [dp_deg, mp_deg])
            layer.mlp.dense_4h_to_h.map(dmp_mesh)

            layer.post_attention_layernorm.split(0, dp_deg)
            layer.post_attention_layernorm.map(dmp_mesh)
            if args.zero >= 2:
                items = ['weight_grad', 'weight'] if args.zero >= 3 else ['weight_grad']
                for key in items:
                    layer.input_layernorm.split(0, dp_deg, item=key)
                    layer.input_layernorm.map(dmp_mesh, item=key)

                    layer.attention.qkv.split(0, dp_mesh.size, item=key)
                    layer.attention.qkv.map(dmp_mesh.transpose().reshape(-1), item=key)
                    layer.attention.dense.split([0, 1], [dp_deg, mp_deg], item=key)
                    layer.attention.dense.map(dmp_mesh, item=key)

                    layer.mlp.dense_h_to_4h.split(0, dp_mesh.size, item=key)
                    layer.mlp.dense_h_to_4h.map(dmp_mesh.transpose().reshape(-1), item=key)
                    layer.mlp.dense_4h_to_h.split([0, 1], [dp_deg, mp_deg], item=key)
                    layer.mlp.dense_4h_to_h.map(dmp_mesh, item=key)

                    layer.post_attention_layernorm.split(0, dp_deg, item=key)
                    layer.post_attention_layernorm.map(dmp_mesh, item=key)

                items = ['bias_grad', 'bias'] if args.zero >= 3 else ['bias_grad']
                for key in items:
                    layer.input_layernorm.split(0, dp_deg, item=key)
                    layer.input_layernorm.map(dmp_mesh, item=key)

                    layer.attention.qkv.split(0, dp_mesh.size, item=key)
                    layer.attention.qkv.map(dmp_mesh.transpose().reshape(-1), item=key)
                    # layer.attention.dense.split(0, dp_deg, item=key)
                    # layer.attention.dense.map(dmp_mesh, item=key)

                    layer.mlp.dense_h_to_4h.split(0, dp_mesh.size, item=key)
                    layer.mlp.dense_h_to_4h.map(dmp_mesh.transpose().reshape(-1), item=key)
                    # layer.mlp.dense_4h_to_h.split(0, dp_deg, item=key)
                    # layer.mlp.dense_4h_to_h.map(dmp_mesh, item=key)

                    layer.post_attention_layernorm.split(0, dp_deg, item=key)
                    layer.post_attention_layernorm.map(dmp_mesh, item=key)
        if args.zero >= 1:
            for opt in layer.input_layernorm.optimizer_ops:
                opt.split(0, dp_deg)
                opt.map(dmp_mesh)
            for opt in layer.attention.qkv.optimizer_ops:
                opt.split(0, dp_mesh.size)
                opt.map(dmp_mesh.transpose().reshape(-1))
            for i, opt in enumerate(layer.attention.dense.optimizer_ops):
                if i == 0:
                    opt.split([0, 1], [dp_deg, mp_deg])
                    opt.map(dmp_mesh)
                else:
                    opt.split(0, dp_deg)
                    opt.map(dmp_mesh)
            for opt in layer.mlp.dense_h_to_4h.optimizer_ops:
                opt.split(0, dp_mesh.size)
                opt.map(dmp_mesh.transpose().reshape(-1))
            for i, opt in enumerate(layer.mlp.dense_4h_to_h.optimizer_ops):
                if i == 0:
                    opt.split([0, 1], [dp_deg, mp_deg])
                    opt.map(dmp_mesh)
                else:
                    opt.split(0, dp_deg)
                    opt.map(dmp_mesh)
            for opt in layer.post_attention_layernorm.optimizer_ops:
                opt.split(0, dp_deg)
                opt.map(dmp_mesh)
        if args.checkpoint:
            layer.recompute()

    if args.ps == 'megatron':
        assert pp_deg == 1
        dp_mesh = dev_topo.create_mesh((ndev, ))
        dmp_mesh = dev_topo.create_mesh((dp_deg, mp_deg), type='gpu')
        stree.root.map(dp_mesh)

        stree.root.embedding.word_embeddings.split([0, 2], [dp_deg, mp_deg])
        stree.root.embedding.word_embeddings.map(dmp_mesh)
        stree.root.embedding.word_embeddings.split(0, dp_deg, item='in:0')
        stree.root.embedding.word_embeddings.map(dmp_mesh, item='in:0')
        stree.root.embedding.position_embeddings.split(0, dp_deg)
        stree.root.embedding.position_embeddings.map(dmp_mesh)
        if args.zero >= 1:
            for i, opt in enumerate(stree.root.embedding.word_embeddings.optimizer_ops):
                opt.split(0, dp_deg * mp_deg)
                opt.map(dmp_mesh.transpose().reshape(-1))
            for i, opt in enumerate(stree.root.embedding.position_embeddings.optimizer_ops):
                opt.split(0, dp_deg)
                opt.map(dmp_mesh)

        if args.zero >= 2:
            items = ['weight_grad', 'weight'] if args.zero >= 3 else ['weight_grad']
            for key in items:
                stree.root.embedding.word_embeddings.split(0, dp_mesh.size, item=key)
                stree.root.embedding.word_embeddings.map(dmp_mesh.transpose().reshape(-1), item=key)
                stree.root.embedding.position_embeddings.split(0, dp_deg, item=key)
                stree.root.embedding.position_embeddings.map(dmp_mesh, item=key)

        for i in range(num_layers):
            split_transformer_layer(stree.root.children[str(i)], dmp_mesh)

        # split final embedding linear
        _seq = num_layers + 1 + (2 if args.seq_first else 0)
        getattr(stree.root, f'seq{_seq}').split([0, 2], [dp_deg, mp_deg])
        getattr(stree.root, f'seq{_seq}').map(dmp_mesh)
        stree.root.criterion.split(0, 1, item='in:1')
        stree.root.criterion.map(dp_mesh, item='in:1')
        stree.schedule()

    if args.ps == 'pp':
        nlayer_per_stage = num_layers // pp_deg
        assert num_layers % pp_deg == 0

        stage_0_mesh = mesh[0]
        stree.root.embedding.map(stage_0_mesh)
        stree.root.embedding.word_embeddings.split([0, 2], [dp_deg, mp_deg])
        stree.root.embedding.word_embeddings.map(stage_0_mesh)
        stree.root.embedding.word_embeddings.split(0, dp_deg, item='in:0')
        stree.root.embedding.word_embeddings.map(stage_0_mesh, item='in:0')
        stree.root.embedding.position_embeddings.split(0, dp_deg)
        stree.root.embedding.position_embeddings.map(stage_0_mesh)
        if args.seq_first:
            stree.root.seq1.map(stage_0_mesh)

        for i in range(num_layers):
            stage_id = i // nlayer_per_stage
            stage_mesh = mesh[stage_id]
            stree.root.children[str(i)].map(stage_mesh)
            split_transformer_layer(stree.root.children[str(i)], stage_mesh)

            nlayer_stage_id = min(i + 1, num_layers - 1) // nlayer_per_stage
            nstage_mesh = mesh[nlayer_stage_id]
            if args.seq_first:
                stree.root.children[str(i)].mlp_add.split(1, dp_deg, item='out')
                stree.root.children[str(i)].mlp_add.map(nstage_mesh, item='out')
                stree.root.children[str(i)].mlp_add.split(1, dp_deg, item='out_grad')
                stree.root.children[str(i)].mlp_add.map(stage_mesh, item='out_grad')
            else:
                stree.root.children[str(i)].mlp_add.split(0, dp_deg, item='out')
                stree.root.children[str(i)].mlp_add.map(nstage_mesh, item='out')
                stree.root.children[str(i)].mlp_add.split(0, dp_deg, item='out_grad')
                stree.root.children[str(i)].mlp_add.map(stage_mesh, item='out_grad')

        # split final embedding linear
        _seq = num_layers + 1 + (2 if args.seq_first else 0)
        getattr(stree.root, f'seq{_seq}').split([0, 2], [dp_deg, mp_deg])
        getattr(stree.root, f'seq{_seq}').map(stage_mesh)
        dp_mesh = stage_mesh.reshape(-1)
        stree.root.criterion.split(0, 1, item='in:1')
        stree.root.criterion.map(dp_mesh, item='in:1')

        input_mesh = [mesh[i].reshape(mesh[i].shape + (1, )) for i in range(pp_deg)]
        input_mesh = np.concatenate(input_mesh, -1)
        stree.root.children['0'].attention.scaled_attention_mask.split([0, 1], [dp_deg, mp_deg], item='in:1')
        stree.root.children['0'].attention.scaled_attention_mask.map(input_mesh, item='in:1')
        embd_mesh = np.concatenate([mesh[0], mesh[-1]], 0).transpose()
        stree.root.embedding.word_embeddings.split(0, mp_deg, item='weight')
        stree.root.embedding.word_embeddings.map(embd_mesh, item='weight')
        ongoing = list(reversed(range(1, pp_deg + 1)))
        stree.schedule(n_macro_batch=args.n_macro_batch,
                       max_ongoing_macro_batch=ongoing)

    # if args.ps in ['zero', 'megatron_zero']:
    #     if args.ps == 'zero':
    #         cube = [[[0, 1, 2, 3, 4, 5, 6, 7]]]
    #         dev_cube = DeviceCube(list(range(8)), cube)
    #         model.split(dev_cube, parts=1)
    #         for name, module in model.named_modules():
    #             if isinstance(module, nn.BuiltinModule):
    #                 module.split(0, 'MP')
    #     MLP.split('dense_h_to_4h', 0, 'DP', 'weight_grad')
    #     MLP.split('dense_h_to_4h', 0, 'DP', 'bias_grad')
    #     MLP.split('dense_h_to_4h', 0, 'DP', 'weight_grad')
    #     MLP.split('dense_h_to_4h', 0, 'DP', 'bias_grad')
    #     Attention.split('qkv', 0, 'DP', 'weight_grad')
    #     Attention.split('qkv', 0, 'DP', 'bias_grad')
    #     Attention.split('dense', 0, 'DP', 'weight_grad')
    #     Attention.split('dense', 0, 'DP', 'bias_grad')

    # if args.ndev == 8:
    #     dev_topo = build_dev_topo()
    # else:
    #     dev_topo = build_dev_topo_N()
    # graph = ProteusModel(dev_topo, train=not args.test)
    # x = graph.Placeholder((32, 128, 1024))
    # label = graph.Placeholder((32, ))

    # # real graph building process
    # y = model(graph, x, None)
    # if not args.test:
    #     criterion(graph, y, label)
    #     optimizer.step(graph)
    if args.disable_collective:
        stree.disable_collective_comm()
    stree.set_bucket_size(args.bucket_size)
    stree.disable_gradient_overlap()

    graph.init_config(stree)
    stree.propagate(graph)

    if args.pp_deg > 1:
        # set share weight
        graph.set_share_weight([
            stree.root.embedding.word_embeddings.op,
            getattr(stree.root, f'seq{num_layers + 1 + 2}').op
        ], stree)

    graph.propagate({})
    graph.symmetric_forward_backward()
    graph.export_config('config.txt')

    stree.dump_tree(config=False)

    if 'titan' in args.cluster or '1080' in args.cluster:
        overlap_factor = 0.3
    else:
        overlap_factor = 0.1 #0.3
    if args.flexflow:
        overlap_factor = 0
    sim = Simulator(graph,
                    stree,
                    cost_type='profile',
                    reprofile=args.reprofile,
                    profile_iters=args.profile_iters,
                    optimizer_overlap=False,
                    share_bandwidth=(not args.no_share_bandwidth) and (not args.flexflow),
                    overlap_factor=overlap_factor,
                    megatron=True,
                    FlexFlow=args.flexflow)
    print('Begin to run simulation...')
    cost = sim.run('log/trace')

    sim.print_stats()
