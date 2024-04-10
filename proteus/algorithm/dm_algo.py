import math
import time
import random
import hashlib
import numpy as np
from collections import defaultdict
from graphviz import Digraph

from proteus import IterType, MapType, OpType, enum_to_str
from proteus.utils import get_strides, flat_to_coordinate
from proteus.strategy import DeviceTopo
from proteus.ir import ProteusModel
from .base import BaseAlgo

factor_dict = {}


def get_factor(n):
    if n not in factor_dict.keys():
        factors = []
        for i in range(1, n + 1):
            if n % i == 0 and (i % 2 == 0 or i == 1):
                factors.append(i)
        factor_dict[n] = factors
    return random.choice(factor_dict[n])


class MNode(object):
    id = 0

    def __init__(self, op, config):
        super().__init__()
        self.op_id = op.id
        self.type = op.type
        self.config = config

        self.part_dims = []
        self.dim_max_parts = {}
        for i, (itype,
                irange) in enumerate(zip(op.iter_space.iters, config.ranges)):
            if itype != IterType.OPAQUE:
                self.part_dims.append((i, irange.end - 1))
                self.dim_max_parts[i] = irange.end - 1
            if i == 0:
                assert itype == IterType.BATCH

        self.id = MNode.id
        MNode.id += 1

        self.next = defaultdict(set)

    def add_next(self, out_id, node_id, in_id):
        self.next[out_id].add((node_id, in_id))

    def __len__(self):
        return len(self.part_dims)


def to_graphviz(nodes):
    graph = Digraph()
    for k, v in nodes.items():
        graph.node(str(v.id), str(k) + ':' + enum_to_str(OpType, v.type))
    for k, v in nodes.items():
        for oid, edges in v.next.items():
            for node_id, in_id in edges:
                graph.edge(str(v.id), str(node_id))
    with open('graph_tree_signodes.txt', 'w') as f:
        f.write(graph.source)


class DMAlgo(BaseAlgo):
    def __init__(self, graph: ProteusModel, dev_topo: DeviceTopo, domain=None, use_is=False):
        super().__init__(dev_topo, ('dp', 'mp'))
        self.graph = graph
        self.use_is = use_is
        self.domain = domain

        self.select_signodes()

    def select_signodes(self):
        # extract significant parallel component
        fops = set(self.graph.ops.keys())
        for _, gid in self.graph.fwop_map.items():
            fops.remove(gid)
        self.nodes = {}
        for op_id in fops:
            op = self.graph.ops[op_id]
            if op.comp_mem_ratio() > 50:
                mnode = MNode(op, self.graph.op_config[op_id])
                self.nodes[op_id] = mnode

        topo_order, out_edges = self.graph.get_linear_graph()
        in_edges = {v[0]: (k, *v[1:]) for k, v in out_edges.items()}
        sig_order = sorted(list(self.nodes),
                           key=lambda op_id: topo_order.index(op_id))
        sig_edges = {}
        for i in range(len(sig_order) - 1):
            prv, nxt = sig_order[i], sig_order[i + 1]
            sig_edges[(prv, nxt)] = (out_edges[prv][1], in_edges[nxt][2])
            self.nodes[prv].add_next(out_edges[prv][1], self.nodes[nxt].id,
                                     in_edges[nxt][2])

        # select sig nodes
        hash_edges = defaultdict(list)
        for (prv, nxt), (out_id, in_id) in sig_edges.items():
            string1 = '{},{},{},{}'.format(self.nodes[prv].type, out_id,
                                           self.graph.ops[prv].ins[0],
                                           self.graph.ops[prv].outs[out_id])
            string2 = '{},{},{},{}'.format(self.nodes[nxt].type, in_id,
                                           self.graph.ops[nxt].ins[0],
                                           self.graph.ops[nxt].ins[in_id])
            hash_edges[(string1, string2)].append((prv, nxt))

        self.sig_op_ids = set()
        for _, v in hash_edges.items():
            self.sig_op_ids.update(list(sorted(v)[0]))
        closed = False
        last_id = sorted(self.sig_op_ids)[-1]
        for k in sig_edges.keys():
            if k[0] == last_id:
                closed = True
                break
        self.sig_op_ids = sorted(self.sig_op_ids)[:-1] if closed else sorted(
            self.sig_op_ids)
        self.sig_nodes = [self.nodes[op_id] for op_id in self.sig_op_ids]

        self.sig_map = defaultdict(set)
        for _, v in hash_edges.items():
            src = set([vid for vid, _ in v])
            dst = set([vid for _, vid in v])
            for op_id in self.sig_op_ids:
                if op_id in src:
                    self.sig_map[op_id].update(src)
                if op_id in dst:
                    self.sig_map[op_id].update(dst)

        to_graphviz(self.nodes)
        if not self.use_is:
            self.sig_nodes = [self.nodes[op_id] for op_id in sig_order]

    def propagate_sig_config(self, opconfig):
        if self.use_is:
            ext_cfg = {}
            for k, cfg in opconfig.items():
                if k in self.sig_map.keys():
                    for op_id in self.sig_map[k]:
                        ext_cfg[op_id] = cfg
            return ext_cfg
        return opconfig

    def optimize(self, type=None, **kwargs):
        if type == 'MCMC':
            if self.domain == 'DMP':
                return self.dmp_mcmc_optimize(**kwargs)
            else:
                return self.mcmc_optimize(**kwargs)
        else:
            return self.brute_force_optimize()

    def brute_force_optimize(self):
        self.visited_hash = defaultdict(lambda: False)

        meshes = []
        for dp in range(1, self.dev_topo.ndevs + 1):
            if self.dev_topo.ndevs % dp == 0:
                shape = [dp, self.dev_topo.ndevs // dp]
                meshes.append(
                    np.array(range(self.dev_topo.ndevs),
                             dtype=np.int).reshape(shape))

        self.cnt = 0
        best_config, min_cost = None, math.inf
        for i, mesh in enumerate(meshes):
            config, cost = self.search_over_mesh(mesh, i)
            if cost < min_cost:
                best_config = config
                min_cost = cost
            print('[{}/{}] searching over {} configs... current min '
                  'cost: {:.2f}'.format(i + 1, len(meshes), self.cnt, min_cost))
        return best_config, min_cost

    def search_over_mesh(self, mesh: np.array, k):
        best_config, min_cost = None, math.inf

        def next_config():
            shape = [len(snode) * 2 - 1 for snode in self.sig_nodes]
            strides = get_strides(shape)
            for k in range(np.prod(shape)):
                cur_config = flat_to_coordinate(k, strides)
                satisfy = True
                for c, snode in zip(cur_config, self.sig_nodes):
                    if (c + 1) % 2 == 1 and snode.part_dims[(c + 1) //
                                                            2][1] < mesh.size:
                        satisfy = False
                        break
                    elif (c + 1) % 2 == 0 and snode.part_dims[
                        (c + 1) // 2][1] < mesh.shape[1]:
                        satisfy = False
                        break
                if not satisfy:
                    continue
                yield cur_config

        for cur_config in next_config():
            encoded = hashlib.md5(str(cur_config).encode('utf-8')).hexdigest()
            if self.visited_hash[encoded]:
                continue
            self.visited_hash[encoded] = True
            self.cnt += 1
            if (self.cnt + 1) % 1000 == 0:
                print('[{}]: cost {:.2f}, config: {}'.format(
                    self.cnt, min_cost, best_config))

            opconfig = {}
            for d, snode in zip(cur_config, self.sig_nodes):
                if (d + 1) % 2 == 1:
                    partition = {snode.part_dims[(d + 1) // 2][0]: mesh.size}
                else:
                    partition = {
                        0: mesh.shape[0],
                        snode.part_dims[(d + 1) // 2][0]: mesh.shape[1]
                    }
                mapdevs = mesh.reshape(-1).tolist()
                opconfig[snode.op_id] = {
                    'partition': partition,
                    'map': (MapType.SHARD, tuple(mapdevs))
                }
            propagated_opcfg = self.propagate_sig_config(opconfig)
            self.graph.propagate(propagated_opcfg)
            self.graph.simulate()
            ret = self.graph.task_manager.evaluate_strategy()
            if ret[0] < min_cost:
                best_config = opconfig.copy()
                min_cost = ret[0]
                print('[{}]: cost {:.2f}, config: {}'.format(
                    self.cnt, min_cost, best_config))
        return best_config, min_cost

    def mcmc_optimize(self, niter=10000, beta=0.1):
        meshes = []
        for dp in range(self.dev_topo.ndevs, 0, -1):
            if self.dev_topo.ndevs % dp == 0:
                shape = [dp, self.dev_topo.ndevs // dp]
                meshes.append(
                    np.array(range(self.dev_topo.ndevs),
                             dtype=np.int).reshape(shape))

        cur_config = {}
        partition = {
            'partition': {
                0: meshes[0].shape[0]
            },
            'map': (MapType.SHARD, tuple(meshes[0].reshape(-1).tolist()))
        }
        for snode in self.sig_nodes:
            cur_config[snode.op_id] = partition.copy()
        propagated_opcfg = self.propagate_sig_config(cur_config)
        # print(propagated_opcfg)
        self.graph.propagate(propagated_opcfg)
        self.graph.simulate()
        ret = self.graph.task_manager.evaluate_strategy()
        cur_cost = ret[0]
        task_ret = ret

        def random_coin(p):
            if random.uniform(0, 1) >= p:
                return False
            return True

        def softmax(x):
            allx = list(x.values())
            fx = {}
            for k, v in x.items():
                fx[k] = np.exp(v) / np.sum(np.exp(allx))
            return fx

        def propose_new_config(opconfig):
            cost = {}
            for k in opconfig.keys():
                st, ed = math.inf, 0
                for _, tasks in self.graph.ops[k].sub_tasks.items():
                    st_l = [
                        self.graph.task_manager.tasks[tsk].start
                        for tsk in tasks
                    ]
                    ed_l = [
                        self.graph.task_manager.tasks[tsk].end for tsk in tasks
                    ]
                    st = min(st_l + [st])
                    ed = max(ed_l + [ed])
                cost[k] = (ed - st) / self.graph.ops[k].get_flops()
            prob = softmax(cost)
            op_id = np.random.choice(list(prob.keys()), p=list(prob.values()))
            # op_id = random.choice(list(opconfig.keys()))
            if random.uniform(0, 1) <= 0.5:
                mesh = random.choice(meshes)
                dim = []
                for dm in opconfig[op_id]['partition'].keys():
                    if dm != 0:
                        dim.append(dm)
                if len(dim) == 0 or self.nodes[op_id].dim_max_parts[
                        dim[0]] < mesh.shape[1]:
                    dim = random.choice(self.nodes[op_id].part_dims[1:])
                    while dim[1] < mesh.shape[1]:
                        dim = random.choice(self.nodes[op_id].part_dims[1:])
            else:
                dp = opconfig[op_id]['partition'][0]
                mesh = np.array(range(self.dev_topo.ndevs),
                                dtype=np.int).reshape(
                                    [dp, self.dev_topo.ndevs // dp])
                dim = random.choice(self.nodes[op_id].part_dims[1:])
                while dim[1] < mesh.shape[1]:
                    dim = random.choice(self.nodes[op_id].part_dims[1:])

            partition = {
                'partition': {
                    0: mesh.shape[0],
                    dim[0]: mesh.shape[1]
                },
                'map': (MapType.SHARD, tuple(mesh.reshape(-1).tolist()))
            }
            newcfg = opconfig.copy()
            newcfg[op_id] = partition
            return newcfg

        best_config, min_cost, citer = cur_config.copy(), cur_cost, 0
        print('Initial: cost {:.2f}, config: {}, {}'.format(min_cost, best_config, task_ret))
        for i in range(niter):
            proposed_cfg = propose_new_config(cur_config)
            propagated_opcfg = self.propagate_sig_config(proposed_cfg)
            self.graph.propagate(propagated_opcfg)
            self.graph.simulate()
            ret = self.graph.task_manager.evaluate_strategy()
            if i == 5:
                exit()
            proposed_cost = ret[0]
            accept_prob = min(1, math.exp((cur_cost - proposed_cost) * beta))
            if random_coin(accept_prob):
                cur_config = proposed_cfg
                cur_cost = proposed_cost
            if proposed_cost < min_cost:
                best_config = proposed_cfg.copy()
                min_cost = proposed_cost
                task_ret = ret
                citer = i
                print('[{}/{}]: cost {:.2f}, config: {}, {}'.format(
                    i, niter, min_cost, best_config, task_ret))
            if (i + 1) % 1000 == 0:
                print('[{}/{}]: cost {:.2f}, config: {}, {}'.format(
                    i, niter, min_cost, best_config, task_ret))
            if i - citer > 200:
                break
        return best_config, min_cost, task_ret

    def dmp_mcmc_optimize(self, niter=10000, beta=0.1):
        meshes = []
        for dp in range(self.dev_topo.ndevs, 0, -1):
            if self.dev_topo.ndevs % dp == 0:
                shape = [dp, self.dev_topo.ndevs // dp]
                meshes.append(
                    np.array(range(self.dev_topo.ndevs),
                             dtype=np.int).reshape(shape))
        macro_batch = 8
        cur_config = {'macro_batch': macro_batch}
        for dev in range(self.dev_topo.ndevs):
            length = len(self.sig_nodes) // self.dev_topo.ndevs
            st = length * dev
            if len(self.sig_nodes) % self.dev_topo.ndevs > dev:
                st = st + dev
                length += 1
            else:
                st = st + len(self.sig_nodes) % self.dev_topo.ndevs
            partition = {
                'partition': {
                    0: macro_batch,
                },
                'map': (MapType.SHARD, tuple([dev] * macro_batch)),
                'devs': [dev]
            }
            for snode in self.sig_nodes[st:st + length]:
                cur_config[snode.op_id] = partition.copy()
        propagated_opcfg = self.propagate_sig_config(cur_config)
        # print(propagated_opcfg)
        self.graph.propagate(propagated_opcfg)
        self.graph.simulate()
        ret = self.graph.task_manager.evaluate_strategy()
        cur_cost = ret[0]
        task_ret = ret

        def random_coin(p):
            if random.uniform(0, 1) >= p:
                return False
            return True

        def softmax(x):
            allx = list(x.values())
            fx = {}
            for k, v in x.items():
                fx[k] = np.exp(v) / np.sum(np.exp(allx))
            return fx

        def propose_new_config(opconfig):
            cost = {}
            for k in opconfig.keys():
                if k == 'macro_batch':
                    continue
                st, ed = math.inf, 0
                for _, tasks in self.graph.ops[k].sub_tasks.items():
                    st_l = [
                        self.graph.task_manager.tasks[tsk].start
                        for tsk in tasks
                    ]
                    ed_l = [
                        self.graph.task_manager.tasks[tsk].end for tsk in tasks
                    ]
                    st = min(st_l + [st])
                    ed = max(ed_l + [ed])
                cost[k] = (ed - st) / self.graph.ops[k].get_flops()
            prob = softmax(cost)
            op_id = np.random.choice(list(prob.keys()), p=list(prob.values()))
            # op_id = random.choice(list(opconfig.keys()))
            type_ = random.choice(['dev', 'deg', 'dim', 'macro'])
            if type_ == 'macro':
                org = opconfig['macro_batch']
                macro = random.choice(list(range(1 - org // 2, org // 2))) + org
                new_cfg = {'macro_batch': macro}
                for k, v in opconfig.items():
                    if k == 'macro_batch':
                        continue
                    new_cfg[k] = {
                        'partition': v['partition'].copy(),
                        'map': (MapType.SHARD, tuple(v['devs'] * macro)),
                        'devs': v['devs'].copy()
                    }
                    new_cfg[k]['partition'][
                        0] = v['partition'][0] * macro // org
                return new_cfg
            elif type_ == 'dim':
                dpdeg = opconfig[op_id]['partition'][0] // opconfig[
                    'macro_batch']
                mpdeg = np.prod(list(opconfig[op_id]['partition'].values()))
                mpdeg = mpdeg // opconfig[op_id]['partition'][0]
                dim = random.choice(self.nodes[op_id].part_dims[1:])
                cnt = 0
                while dim[1] < mpdeg:
                    cnt += 1
                    dim = random.choice(self.nodes[op_id].part_dims[1:])
                    if cnt > 10:
                        dpdeg = dpdeg * mpdeg
                        mpdeg = 1
                        break
                partition = {
                    'partition': {
                        0: dpdeg * opconfig['macro_batch'],
                        dim[0]: mpdeg
                    },
                    'map': opconfig[op_id]['map'],
                    'devs': opconfig[op_id]['devs']
                }
            elif type_ == 'deg':
                dp = get_factor(len(opconfig[op_id]['devs']))
                mesh = np.array(opconfig[op_id]['devs'],
                                dtype=np.int).reshape(dp, -1)
                dim = []
                for dm in opconfig[op_id]['partition'].keys():
                    if dm != 0:
                        dim.append(dm)
                if len(dim) == 0 or self.nodes[op_id].dim_max_parts[
                        dim[0]] < mesh.shape[1]:
                    dim = random.choice(self.nodes[op_id].part_dims[1:])
                    while dim[1] < mesh.shape[1]:
                        dim = random.choice(self.nodes[op_id].part_dims[1:])
                partition = {
                    'partition': {
                        0: mesh.shape[0] * opconfig['macro_batch'],
                        dim[0]: mesh.shape[1]
                    },
                    'map':
                    (MapType.SHARD,
                     tuple(mesh.reshape(-1).tolist() *
                           opconfig['macro_batch'])),
                    'devs':
                    opconfig[op_id]['devs']
                }
            else:
                org_length = len(opconfig[op_id]['devs'])
                dpdeg = opconfig[op_id]['partition'][0] // opconfig[
                    'macro_batch']
                mpdeg = np.prod(list(opconfig[op_id]['partition'].values()))
                mpdeg = mpdeg // opconfig[op_id]['partition'][0]
                mpdim = set(opconfig[op_id]['partition'].keys()).remove(0)
                mpdim = 1 if mpdim is None or len(mpdim) == 0 else list(
                    mpdim)[0]
                rl = random.choice(['left', 'right'])
                move = random.choice(['increase', 'shift', 'reduce'])
                dp_or_mp = random.choice(['dp', 'mp'])
                if dp_or_mp == 'dp':
                    num = mpdeg
                else:
                    num = dpdeg

                if move == 'increase':
                    length = org_length + num
                    if length > self.dev_topo.ndevs:
                        length = org_length
                        num = 0
                    if rl == 'left':
                        st = opconfig[op_id]['devs'][
                            0] + 2 * self.dev_topo.ndevs - num
                        dev_help = list(range(self.dev_topo.ndevs)) * 5
                        devs = dev_help[st:st + length]
                    elif rl == 'right':
                        st = opconfig[op_id]['devs'][0] + 2 * self.dev_topo.ndevs
                        dev_help = list(range(self.dev_topo.ndevs)) * 5
                        devs = dev_help[st:st + length]
                    if dp_or_mp == 'dp':
                        dpdeg = dpdeg if length == org_length else dpdeg + 1
                    else:
                        mpdeg = mpdeg if length == org_length else mpdeg + 1
                elif move == 'reduce':
                    length = num if org_length == num else org_length - num
                    if rl == 'left':
                        devs = opconfig[op_id]['devs'][org_length - length:]
                    elif rl == 'right':
                        devs = opconfig[op_id]['devs'][:length]
                    if dp_or_mp == 'dp':
                        dpdeg = dpdeg if dpdeg == 1 else dpdeg - 1
                    else:
                        mpdeg = mpdeg if mpdeg == 1 else mpdeg - 1
                else:
                    length = org_length
                    if rl == 'left':
                        st = opconfig[op_id]['devs'][
                            0] + 2 * self.dev_topo.ndevs - num
                        dev_help = list(range(self.dev_topo.ndevs)) * 5
                        devs = dev_help[st:st + length]
                    elif rl == 'right':
                        st = opconfig[op_id]['devs'][
                            0] + 2 * self.dev_topo.ndevs + num
                        dev_help = list(range(self.dev_topo.ndevs)) * 5
                        devs = dev_help[st:st + length]

                partition = {
                    'partition': {
                        0: dpdeg * opconfig['macro_batch'],
                        mpdim: mpdeg
                    },
                    'map':
                    (MapType.SHARD, tuple(devs * opconfig['macro_batch'])),
                    'devs': devs
                }
            newcfg = opconfig.copy()
            newcfg[op_id] = partition
            return newcfg

        best_config, min_cost, citer = cur_config.copy(), cur_cost, 0
        print('Initial: cost {:.2f}, config: {}'.format(min_cost, best_config))
        for i in range(niter):
            proposed_cfg = propose_new_config(cur_config)
            propagated_opcfg = self.propagate_sig_config(proposed_cfg)
            self.graph.propagate(propagated_opcfg)
            self.graph.simulate()
            ret = self.graph.task_manager.evaluate_strategy()
            proposed_cost = ret[0]
            accept_prob = min(1, math.exp((cur_cost - proposed_cost) * beta))
            if random_coin(accept_prob):
                cur_config = proposed_cfg
                cur_cost = proposed_cost
            if proposed_cost < min_cost:
                best_config = proposed_cfg.copy()
                min_cost = proposed_cost
                task_ret = ret
                citer = i
                print('[{}/{}]: cost {:.2f}, config: {}'.format(
                    i, niter, min_cost, best_config))
            if (i + 1) % 1000 == 0:
                print('[{}/{}]: cost {:.2f}, config: {}'.format(
                    i, niter, min_cost, best_config))
            if i - citer > 200:
                break
        return best_config, min_cost, task_ret

    def genetic_optimize(self):
        pass
