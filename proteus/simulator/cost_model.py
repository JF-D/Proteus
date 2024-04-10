import os
import shutil
import json
import math
import bisect
from collections import defaultdict
import proteus.binding as binding

global PROFILE_ITERS
PROFILE_ITERS = 10

global _communicators
_communicators = {}


def set_profile_iters(niters):
    global PROFILE_ITERS
    PROFILE_ITERS = niters


def get_profile_iters():
    global PROFILE_ITERS
    return PROFILE_ITERS


stats = {
    7: [
        15.679635107517242, 15.075094997882843, 15.474595129489899,
        15.021823346614838, 14.860071241855621, 15.053004026412964,
        14.66110348701477, 14.85627144575119, 14.84949141740799,
        14.803856611251831, 15.12765884399414, 15.169419348239899,
        14.901086688041687, 15.298984944820404, 20.679086446762085,
        27.793273329734802, 35.73175519704819, 40.592700242996216,
        59.14319306612015, 88.55435997247696, 144.66099441051483,
        281.3863754272461, 517.9806426167488, 1018.6421871185303,
        2022.9560136795044, 4034.3500673770905, 8056.372031569481,
        16092.208549380302, 32151.660285890102, 64279.94839847088
    ],
    25: [
        15.863478183746338, 22.258572280406952, 23.429282009601593,
        21.52036875486374, 24.150162935256958, 21.517425775527954,
        16.352608799934387, 20.895563066005707, 21.76128327846527,
        20.03271132707596, 20.436681807041168, 20.068474113941193,
        20.14636993408203, 20.323097705841064, 20.585618913173676,
        20.47218382358551, 24.34913069009781, 29.548555612564087,
        35.48402339220047, 47.88108170032501, 70.80391049385071,
        117.05726385116577, 207.89731293916702, 392.71030575037,
        759.1724023222923, 1492.8526803851128, 2955.4424434900284,
        6025.828905403614, 11742.177046835423, 23457.500860095024
    ],
    50: [
        15.930160880088806, 14.998391270637512, 15.359483659267426,
        15.031322836875916, 14.540404081344604, 14.609098434448242,
        14.87724483013153, 14.744400978088379, 14.74667340517044,
        14.939755201339722, 15.292763710021973, 14.742538332939148,
        15.184283256530762, 15.59622585773468, 15.389733016490936,
        16.33238047361374, 21.350309252738953, 24.42728728055954,
        32.480619847774506, 37.87528723478317, 49.21134561300278,
        72.37110286951065, 118.13826858997345, 211.18368953466415,
        394.7889432311058, 776.3361930847168, 1493.123210966587,
        2957.04897493124, 5886.068902909756, 11744.802109897137
    ]
}

datasize = [2**i for i in range(30)]


def comm_p2p(MBytes, bw_GB_per_sec):
    # comm_time(ms) = latency + transfer_time
    if bw_GB_per_sec in stats:
        idx = bisect.bisect_left(datasize, MBytes * 1e6)
        latency = stats[bw_GB_per_sec]
        ms = ((MBytes * 1e6 - datasize[idx - 1]) * latency[idx] +
              (datasize[idx] - MBytes * 1e6) * latency[idx - 1]) / (
                  datasize[idx] - datasize[idx - 1]) / 1e3
        return ms
    ms = 0.005 + MBytes / bw_GB_per_sec
    return ms


def comm_all_reduce(MBytes, bw_GB_per_sec, group_size):
    mbytes = MBytes * 1e6 / group_size
    if bw_GB_per_sec in stats:
        idx = bisect.bisect_left(datasize, mbytes)
        latency = stats[bw_GB_per_sec]
        ms = ((mbytes - datasize[idx - 1]) * latency[idx] +
              (datasize[idx] - mbytes) * latency[idx - 1]) / (
                  datasize[idx] - datasize[idx - 1]) / 1e3
        return ms * 2 * (group_size - 1)
    ms = 0.005 + MBytes / group_size / bw_GB_per_sec
    return ms * 2 * (group_size - 1)


def comm_reduce_scatter(MBytes, bw_GB_per_sec, group_size):
    mbytes = MBytes * 1e6 / group_size
    if bw_GB_per_sec in stats:
        idx = bisect.bisect_left(datasize, mbytes)
        latency = stats[bw_GB_per_sec]
        ms = ((mbytes - datasize[idx - 1]) * latency[idx] +
              (datasize[idx] - mbytes) * latency[idx - 1]) / (
                  datasize[idx] - datasize[idx - 1]) / 1e3
        return ms * (group_size - 1)
    ms = 0.005 + MBytes / group_size / bw_GB_per_sec
    return ms * (group_size - 1)


def comm_all_gather(MBytes, bw_GB_per_sec, group_size):
    mbytes = MBytes * 1e6 / group_size
    if bw_GB_per_sec in stats:
        idx = bisect.bisect_left(datasize, mbytes)
        latency = stats[bw_GB_per_sec]
        ms = ((mbytes - datasize[idx - 1]) * latency[idx] +
              (datasize[idx] - mbytes) * latency[idx - 1]) / (
                  datasize[idx] - datasize[idx - 1]) / 1e3
        return ms * (group_size - 1)
    ms = 0.005 + MBytes / group_size / bw_GB_per_sec
    return ms * (group_size - 1)


class OpCostModel:
    cost_fn = defaultdict(lambda: {})
    cache_key_fn = defaultdict(lambda: None)
    cache = defaultdict(lambda: defaultdict(lambda: {}))
    reprofiled_op = defaultdict(lambda: defaultdict(lambda: {}))
    cluster = None

    def __init__(self, filename='cache'):
        self.filename = f'{filename}.json'
        if os.path.isfile(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    cache = json.load(f)
                    OpCostModel.cache.update(cache)
            except:
                print('[warning] Op cost cache file cannot be load!')

    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(OpCostModel.cache, f)

    @staticmethod
    def register(type, cost_type, func, cache_key_fn=None):
        OpCostModel.cost_fn[type.__name__][cost_type] = func
        if cache_key_fn is not None:
            OpCostModel.cache_key_fn[type.__name__] = cache_key_fn

    @staticmethod
    def measure_op_cost(op, dev, cost_type, reprofile=False, profile_iters=10):
        cost_cache = OpCostModel.cache[op.__class__.__name__][cost_type]
        if OpCostModel.cache_key_fn[op.__class__.__name__]:
            key = OpCostModel.cache_key_fn[op.__class__.__name__](op)
        else:
            key = '{}_{}'.format(op.ins, op.outs)
        reprofiled_op = OpCostModel.reprofiled_op[
            op.__class__.__name__][cost_type]
        reprofile = reprofile and key not in reprofiled_op
        if key in cost_cache and not reprofile:
            return cost_cache[key]
        if cost_type == 'profile':
            set_profile_iters(profile_iters)
        cost = OpCostModel.cost_fn[op.__class__.__name__][cost_type](op, dev)
        print('profile:', op, op.ins, op.outs, f'cost: {cost:.3f}ms')
        cost_cache[key] = cost
        reprofiled_op[key] = True
        return cost

    @staticmethod
    def comm_cost(type, volume, bw, group='default', root=0, FlexFlow=False):
        if type in ['cpu2gpu', 'gpu2cpu']:
            ct = 0.05 + volume / 12
            return ct

        global _communicators
        if group == 'default':
            group = [i for i in range(OpCostModel.cluster.ngpus)]
        key = ''
        sort_group = sorted(group)
        for g in sort_group:
            key = key + str(g) + ','
        if key not in _communicators:
            n_gpu_per_node = OpCostModel.cluster.n_gpu_per_node
            topofile = OpCostModel.cluster.topo_file
            local_rank_groups, group_ranks = [], []
            cgroup, cg_rank = [], []
            cur_node_id = -1
            intra_node_rank = 0
            for g in sort_group:
                if g // n_gpu_per_node == cur_node_id:
                    cgroup.append(g % n_gpu_per_node)
                    cg_rank.append(intra_node_rank)
                    intra_node_rank += 1
                else:
                    if len(cgroup) > 0:
                        local_rank_groups.append(cgroup)
                        group_ranks.append(cg_rank)
                    # init
                    cgroup = []
                    cg_rank = []
                    intra_node_rank = 0
                    cur_node_id = g // n_gpu_per_node
                    # set rank and group
                    cgroup.append(g % n_gpu_per_node)
                    cg_rank.append(intra_node_rank)
                    intra_node_rank += 1
            if len(cgroup) > 0:
                local_rank_groups.append(cgroup)
                group_ranks.append(cg_rank)
            groups_, groups_rank_ = [], []
            group_set = set()
            for lg, gr in zip(local_rank_groups, group_ranks):
                if tuple(lg) not in group_set:
                    group_set.add(tuple(lg))
                    groups_.append(lg)
                    groups_rank_.append(gr)

            _communicators[key] = binding.Communicator(groups_, groups_rank_,
                                                       len(group),
                                                       len(local_rank_groups),
                                                       topofile)

        comm = _communicators[key]
        type_intra = comm.get_graph_type_intra()
        type_inter = comm.get_graph_type_inter()
        cross_node = comm.get_cross_node()

        if type == 'p2p':
            # return comm_p2p(volume, bw)
            ct = comm.broadcast(math.ceil(volume * 1e6), root)
        elif type == 'reduce':
            # return comm_all_reduce(volume, bw, len(group))
            ct = comm.reduce(math.ceil(volume * 1e6), root)
        elif type == 'all_reduce':
            # return comm_all_reduce(volume, bw, len(group))
            ct = comm.allreduce(math.ceil(volume * 1e6))
            if len(group) == 8 and 'titan' in OpCostModel.cluster.topo_file:
                # titanxp all_reduce bandwidth adjust
                ct = (4.2, ct[1])
        elif type == 'all_gather':
            # return comm_all_gather(volume, bw, len(group))
            ct = comm.allgather(math.ceil(volume * 1e6))
        elif type == 'reduce_scatter':
            # return comm_reduce_scatter(volume, bw, len(group))
            ct = comm.reduce_scatter(math.ceil(volume * 1e6))
        elif type == 'scatter' or type == 'gather':
            ct = comm.broadcast(math.ceil(volume * 1e6), root)
        elif type == 'all_to_all':
            ct = comm.broadcast(math.ceil(volume * 1e6), 0)
            if cross_node:
                # low utilization and approximate bandwidth share
                ct = (0.5 * ct[0] / OpCostModel.cluster.n_node, ct[1] * 0.5)
            else:
                ct = (ct[0], ct[1] * 0.1)
        else:
            raise NotImplementedError
        return ct, type_intra, type_inter, cross_node


def register_op_cost_model(op_class, cost_type, func, cache_key_fn=None):
    OpCostModel.register(op_class, cost_type, func, cache_key_fn)
