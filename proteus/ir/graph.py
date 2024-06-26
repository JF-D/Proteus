import queue
import numpy as np
from collections import defaultdict
from graphviz import Digraph
from toposort import toposort_flatten

from proteus import DataType, OpType, MapType, enum_to_str
from proteus.simulator.task_manager import TaskManager
from .device import DeviceTopo
from .tensor import Tensor, Input, Output, Parameter, Gradient, Buffer
from .node import Edge
from .ops import *
from .node_config import TensorConfig, OpConfig


def _pair(x):
    if not isinstance(x, (list, tuple)):
        return (x, x)
    else:
        return x


class Graph:

    def __init__(self):
        self.ops = {}
        self.edges = set()


class ProteusModel(object):

    def __init__(self, train=True):
        super().__init__()
        self.train = train
        self.datas = {}
        self.ops = {}
        self.edges = set()

        self.grad_map = {}
        self.fwop_map = {}
        self.inputs = []
        self.outputs = []
        self._parameters = []
        self.share_parameters = []

        self.forward_graph = Graph()
        self.backward_graph = Graph()
        self.optimizer_graph = Graph()

    def init_config(self, stree):
        self.op_config = {}
        self.data_config = {}
        for k, v in self.datas.items():
            self.data_config[k] = TensorConfig(v,
                                               stree.dev_topo.ndevs,
                                               max_parts=stree.max_parts,
                                               stride=stree.stride)
        for k, v in self.ops.items():
            self.op_config[k] = OpConfig(v,
                                         stree.dev_topo.ndevs,
                                         max_parts=stree.max_parts,
                                         stride=stree.stride)
        return self.data_config, self.op_config

    def symmetric_forward_backward(self):
        for p, g in self.grad_map.items():
            config = self.data_config[p].get_config()
            if not self.data_config[g].manual:
                self.data_config[g].set_config(config)
        for fw, bw in self.fwop_map.items():
            config = self.op_config[fw].get_config()
            if not self.op_config[bw].manual:
                self.op_config[bw].set_config(config)

    def partition_and_map(self, obj, config):
        if isinstance(obj, Tensor):
            self.data_config[obj.id].set_config(config)
        else:
            self.op_config[obj.id].set_config(config)

    def forward_edge_set_to_adj(self, forward=True, edges=None, reverse=False):
        fdatas, fops = set(self.datas.keys()), set(self.ops.keys())
        if forward:
            for _, gid in self.grad_map.items():
                fdatas.remove(gid)
            for _, gid in self.fwop_map.items():
                fops.remove(gid)

        if edges is None:
            edges = self.edges
        e_data, e_op = defaultdict(list), defaultdict(list)
        for e in edges:
            if e.data in fdatas and e.op in fops:
                if e.privilege in ['r', 'rw']:
                    if reverse:
                        e_op[e.op].append((e.data, e.id))
                    else:
                        e_data[e.data].append((e.op, e.id))
                else:
                    if reverse:
                        e_data[e.data].append((e.op, e.id))
                    else:
                        e_op[e.op].append((e.data, e.id))
        return e_data, e_op

    def set_share_weight(self, share_weight_ops, stree):
        # suppose len(share_weight_ops) < 10
        for i, weight in enumerate(share_weight_ops[0].read[1:]):
            dweight = self.datas[self.grad_map[weight.id]]
            self.share_parameters.append(weight.id)
            self.share_parameters.append(dweight.id)
            for index, op in enumerate(share_weight_ops):
                assert op.read[i + 1] == weight
                new_id = weight.id + 0.1 * (index + 1)
                share_w = Tensor(weight.size(),
                                 dtype=weight.dtype,
                                 id=new_id,
                                 requires_grad=weight.requires_grad,
                                 name=weight.name + '_share')
                share_w = Parameter(share_w, requires_grad=True)
                tb_remove = Edge('r', op.id, 1, weight.id)
                assert tb_remove in self.edges
                self.edges.remove(tb_remove)
                self.edges.add(Edge('r', op.id, 1, share_w.id))
                read_list = list(op.read)
                read_list[1] = share_w
                op.read = tuple(read_list)
                share_w.add_consumer(op, 1)

                # deal with gradient
                if op.id not in self.fwop_map:
                    continue
                op_bw = self.ops[self.fwop_map[op.id]]
                new_id = dweight.id + 0.1 * (index + 1)
                share_dw = Tensor(weight.size(),
                                  dtype=weight.dtype,
                                  id=new_id,
                                  requires_grad=False,
                                  name=dweight.name + '_share')
                share_dw = Gradient(share_dw, share_w)
                read_list, write_list = list(op_bw.read), list(op_bw.write)
                e = Edge('r', op_bw.id, 2, weight.id)
                if e in self.edges:
                    self.edges.remove(e)
                    self.edges.add(Edge('r', op_bw.id, 2, share_w.id))
                    read_list[2] = share_w
                    share_w.add_consumer(op_bw, 2)
                for e in self.edges:
                    if e.op == op_bw.id and e.data == dweight.id:
                        self.edges.add(
                            Edge(e.privilege, op_bw.id, e.id, share_dw.id))
                        self.edges.remove(e)
                        if e.privilege in ['r', 'rw']:
                            read_list[e.id] = share_dw
                            share_dw.add_consumer(op_bw, e.id)
                        else:
                            write_list[e.id] = share_dw
                            share_dw.add_producer(op_bw, e.id)
                        continue
                op_bw.read = tuple(read_list)
                op_bw.write = tuple(write_list)

                self.datas[share_w.id] = share_w
                self.datas[share_dw.id] = share_dw
                self.grad_map[share_w.id] = share_dw.id
                weight.share_params.append(share_w)
                dweight.share_params.append(share_dw)
                share_w.share_from = weight
                share_dw.share_from = dweight

                for data in [share_w, share_dw]:
                    self.data_config[data.id] = TensorConfig(
                        data,
                        stree.dev_topo.ndevs,
                        max_parts=stree.max_parts,
                        stride=stree.stride)

    def get_linear_graph(self):
        """Get the longest op path in graph (linear graph).
        """
        next_dict = dict()
        edges, vpair = self.make_op_graph(forward=True, reverse=True)
        topo_order = toposort_flatten(edges)

        next_dict, dist = {}, dict.fromkeys(topo_order, 0)
        for op_id in topo_order[::-1]:
            for prev_op in edges[op_id]:
                if dist[prev_op] < dist[op_id] + 1:
                    dist[prev_op] = dist[op_id] + 1
                    next_dict[prev_op] = op_id

        edges = defaultdict(set)
        prev_id = max(dist, key=dist.get)
        while prev_id in next_dict.keys():
            next_id = next_dict[prev_id]
            in_id, out_id = vpair[(next_id, prev_id)]
            edges[prev_id].add((next_id, out_id, in_id))
            prev_id = next_id
        edges = {k: list(v)[0] for k, v in edges.items()}
        return topo_order, edges

    def make_op_graph(self, forward=False, reverse=False):
        """Convert op-data graph to op graph.

        :param forward: only consider forward graph or not
        :param reverse: op graph in forward mode or reverse mode
        """
        e_data, e_op = self.forward_edge_set_to_adj(forward=forward)
        edges, vpair = defaultdict(set), {}
        for op_id, ndatas in e_op.items():
            for ndata, out_id in ndatas:
                for nop_id, in_id in e_data[ndata]:
                    if reverse:
                        # prev mode
                        edges[nop_id].add(op_id)
                        assert (nop_id, op_id) not in vpair.keys()
                        vpair[(nop_id, op_id)] = (in_id, out_id)
                    else:
                        # next mode
                        edges[op_id].add(nop_id)
                        assert (op_id, nop_id) not in vpair.keys()
                        vpair[(op_id, nop_id)] = (out_id, in_id)
        return edges, vpair

    def propagate(self, opconfig, datacfg={}):
        data_parted = defaultdict(lambda: False)
        op_parted = defaultdict(lambda: False)
        for did in datacfg.keys():
            self.data_config[did].manual_pconfig(datacfg[did])
            data_parted[did] = True

        for op_id, op_cfg in self.op_config.items():
            op_parted[op_id] = op_cfg.manual
        for data_id, data_cfg in self.data_config.items():
            data_parted[data_id] = data_cfg.manual

        def finish():
            ops_, datas_ = [], []
            for k, v in op_parted.items():
                if not v and k not in list(self.fwop_map.values()):
                    ops_.append(self.ops[k])
            for k, v in data_parted.items():
                if not v and not isinstance(self.datas[k], Gradient):
                    datas_.append(self.datas[k])
            return len(ops_) == 0 and len(datas_) == 0

        self.set_depth()
        rounds = 0
        while not finish():
            rounds += 1
            if rounds > 10:
                print('[Error] Propagate error!')
                exit()
            self.propagate_forward(opconfig, datacfg, op_parted, data_parted)
            self.propagate_backward(opconfig, datacfg, op_parted, data_parted)
            self.symmetric_forward_backward()
            self.export_config('log/propagate_config.txt')
        self.export_config('log/propagate_config.txt')

    def set_depth(self):
        q = queue.Queue()
        data_deg, op_deg = defaultdict(int), defaultdict(int)
        share_weight, share_grad = [], []
        for param_id in self.share_parameters:
            param = self.datas[param_id]
            if len(param.producer) == 0:
                share_weight.append(param)
            else:
                for grad in param.share_params:
                    share_grad.append(grad.id)
        for data_id, data in self.datas.items():
            data_deg[data_id] = len(data.producer)
            if data_deg[data_id] == 0 and data_id not in self.share_parameters:
                q.put(('data', data_id))
        for op_id, op in self.ops.items():
            op_deg[op_id] = len(op.read)
            for share_w in share_weight:
                if share_w in op.read:
                    op_deg[op_id] -= 1
            if op_deg[op_id] == 0:
                self.ops[op_id].depth = 0
                q.put(('op', op_id))

        while not q.empty():
            ntype, nid = q.get()
            if ntype == 'op':
                for data in self.ops[nid].write:
                    data_deg[data.id] -= 1
                    if data_deg[data.id] == 0:
                        q.put(('data', data.id))
                    if data.id in share_grad:
                        data_deg[data.share_from.id] -= 1
                        if data_deg[data.share_from.id] == 0:
                            q.put(('data', data.share_from.id))
            else:
                for (op, _) in self.datas[nid].consumer:
                    op_deg[op.id] -= 1
                    if op_deg[op.id] == 0:
                        producer_depth = [-1]
                        for prev_data in op.read:
                            for (prev_op, _) in prev_data.producer:
                                producer_depth.append(prev_op.depth)
                        op.depth = 1 + max(producer_depth)
                        q.put(('op', op.id))

    def propagate_forward(self, opcfg, datacfg, op_parted, data_parted):
        fdatas, fops = set(self.datas.keys()), set(self.ops.keys())
        for _, gid in self.grad_map.items():
            fdatas.remove(gid)
        for _, gid in self.fwop_map.items():
            fops.remove(gid)

        e_data, e_op = defaultdict(list), defaultdict(list)
        data_deg, op_deg = defaultdict(int), defaultdict(int)
        for e in self.edges:
            if e.data in fdatas and e.op in fops:
                if e.privilege in ['r', 'rw']:
                    e_data[e.data].append((e.op, e.id))
                    op_deg[e.op] += 1
                else:
                    e_op[e.op].append((e.data, e.id))
                    data_deg[e.data] += 1

        topo_queue = queue.Queue()
        for did in fdatas:
            if data_deg[did] == 0:
                topo_queue.put(('data', did))
        for op_id in fops:
            if op_deg[op_id] == 0:
                topo_queue.put(('op', op_id))
        while not topo_queue.empty():
            ntype, idx = topo_queue.get()
            if ntype == 'data':
                for op_id, rid in e_data[idx]:
                    if data_parted[idx]:
                        if op_id not in opcfg.keys() and not op_parted[op_id]:
                            self.ops[op_id].set_input_config(
                                rid, self.data_config[idx])
                    op_deg[op_id] -= 1
                    if op_deg[op_id] == 0:
                        if op_parted[op_id]:
                            topo_queue.put(('op', op_id))
                            continue
                        if op_id not in opcfg.keys():
                            status = self.infer_op_from_data(op_id,
                                                             reverse=False)
                            op_parted[op_id] = bool(status)
                        else:
                            self.op_config[op_id].manual_pconfig(opcfg[op_id])
                            op_parted[op_id] = True
                        topo_queue.put(('op', op_id))
            elif ntype == 'op':
                for data_id, wid in e_op[idx]:
                    if op_parted[idx] and not data_parted[data_id]:
                        if data_id not in datacfg.keys():
                            cfg = self.ops[idx].infer_output_config(
                                wid, self.op_config[idx])
                            self.data_config[data_id].set_config(cfg)
                        else:
                            self.data_config[data_id].manual_pconfig(
                                datacfg[data_id])
                        data_parted[data_id] = True
                    data_deg[data_id] -= 1
                    if data_deg[data_id] == 0:
                        topo_queue.put(('data', data_id))
            else:
                assert False, 'Unknown node type {}'.format(ntype)

    def propagate_backward(self, opcfg, datacfg, op_parted, data_parted):
        fdatas, fops = set(self.datas.keys()), set(self.ops.keys())
        for _, gid in self.grad_map.items():
            fdatas.remove(gid)
        for _, gid in self.fwop_map.items():
            fops.remove(gid)

        e_data, e_op = defaultdict(list), defaultdict(list)
        data_deg, op_deg = defaultdict(int), defaultdict(int)
        for e in self.edges:
            if e.data in fdatas and e.op in fops:
                if e.privilege in ['r', 'rw']:
                    e_op[e.op].append((e.data, e.id))
                    data_deg[e.data] += 1
                else:
                    e_data[e.data].append((e.op, e.id))
                    op_deg[e.op] += 1

        topo_queue = queue.Queue()
        for did in fdatas:
            if data_deg[did] == 0:
                topo_queue.put(('data', did))
        for op_id in fops:
            if op_deg[op_id] == 0:
                topo_queue.put(('op', op_id))
        while not topo_queue.empty():
            ntype, idx = topo_queue.get()
            if ntype == 'data':
                for op_id, wid in e_data[idx]:
                    if data_parted[idx] and not op_parted[op_id]:
                        if op_id not in opcfg.keys():
                            self.ops[op_id].set_output_config(
                                wid, self.data_config[idx])
                    op_deg[op_id] -= 1
                    if op_deg[op_id] == 0:
                        if op_parted[op_id]:
                            topo_queue.put(('op', op_id))
                            continue
                        if op_id not in opcfg.keys():
                            status = self.infer_op_from_data(op_id,
                                                             reverse=True)
                            op_parted[op_id] = bool(status)
                        else:
                            self.op_config[op_id].manual_pconfig(opcfg[op_id])
                            op_parted[op_id] = True
                        topo_queue.put(('op', op_id))
            elif ntype == 'op':
                for data_id, rid in e_op[idx]:
                    if op_parted[idx] and not data_parted[data_id]:
                        if data_id not in datacfg.keys():
                            cfg = self.ops[idx].infer_input_config(
                                rid, self.op_config[idx])
                            self.data_config[data_id].set_config(cfg)
                        else:
                            self.data_config[data_id].manual_pconfig(
                                datacfg[data_id])
                        data_parted[data_id] = True
                    data_deg[data_id] -= 1
                    if data_deg[data_id] == 0:
                        topo_queue.put(('data', data_id))
            else:
                assert False, 'Unknown node type {}'.format(ntype)

    def infer_op_from_data(self, op_id, reverse=False):
        '''Infer op partition config from data partition config.
        '''
        if reverse:
            # infer op from output
            cfg = self.ops[op_id].infer_op_from_output()
        else:
            # infer op from input
            cfg = self.ops[op_id].infer_op_from_input()
        if cfg:
            self.op_config[op_id].set_config(cfg)
        return cfg

    def export_config(self, file, shape=True):
        # make the file path valid if not existed
        import os
        if not os.path.exists(os.path.dirname(file)) and os.path.dirname(file) != '':
            print(f"Creating directory {os.path.dirname(file)}")
            os.makedirs(os.path.dirname(file))

        
        with open(file, 'w') as f:
            for k, data in self.datas.items():
                if shape:
                    name = '{:12} {:18}: '.format(data.name + '_' + str(k),
                                                  str(data.shape))
                else:
                    name = '{:12}: '.format(data.name + '_' + str(k))
                f.write(name)
                self.data_config[k].export(f)
            for k, op in self.ops.items():
                name = '{:20}: '.format(str(op.type) + '_' + str(k))
                f.write(name)
                self.op_config[k].export(f)

    def set_op_pconfig(self, ops, pconfig):
        if not isinstance(ops, (tuple, list)):
            ops = [ops]
        for i, op in enumerate(ops):
            self.op_config[op.id].manual_pconfig(pconfig)
            if i == 0:
                # forward op
                for ind, rid in enumerate(op.read):
                    plain_op = not isinstance(op, (SGD, Adam)) and isinstance(
                        rid, (Input, Parameter, Buffer))
                    opt_op = isinstance(op, (SGD, Adam)) and isinstance(
                        rid, Buffer)
                    if (plain_op
                            or opt_op) and not self.data_config[rid.id].manual:
                        cfg = op.infer_input_config(ind, self.op_config[op.id])
                        self.data_config[rid.id].set_config(cfg)
                        self.data_config[rid.id].manual = True and not opt_op
                        if rid.id in self.grad_map.keys():
                            gid = self.grad_map[rid.id]
                            self.data_config[gid].set_config(cfg)
                for oud, wid in enumerate(op.write):
                    if self.op_config[op.id].manual and not self.data_config[
                            wid.id].manual:
                        cfg = op.infer_output_config(oud,
                                                     self.op_config[op.id])
                        self.data_config[wid.id].set_config(cfg)
                        self.data_config[wid.id].manual = True
                        if wid.id in self.grad_map.keys():
                            gid = self.grad_map[wid.id]
                            self.data_config[gid].set_config(cfg)
                # manual partition data
                if len(pconfig['data']) > 0:
                    for param, cfg in pconfig['data'].items():
                        if param == 'weight':
                            tids = [op.read[1].id]
                            if tids[0] in self.grad_map.keys():
                                tids.append(self.grad_map[tids[0]])
                        elif param == 'bias':
                            tids = [op.read[2].id]
                            if tids[0] in self.grad_map.keys():
                                tids.append(self.grad_map[tids[0]])
                        elif param == 'buffer':
                            assert isinstance(op, Adam)
                            tids = [rd.id for rd in op.read[2:]]
                        elif param == 'weight_grad':
                            tids = []
                            if op.read[1].id in self.grad_map.keys():
                                tids.append(self.grad_map[op.read[1].id])
                        elif param == 'bias_grad':
                            tids = []
                            if op.read[2].id in self.grad_map.keys():
                                tids.append(self.grad_map[op.read[2].id])
                        elif param == 'out':
                            for wdata in op.write:
                                self.data_config[wdata.id].manual_pconfig(cfg)
                            continue
                        elif param == 'out_grad':
                            for wdata in op.write:
                                if wdata.id in self.grad_map.keys():
                                    grad_id = self.grad_map[wdata.id]
                                    self.data_config[grad_id].manual_pconfig(
                                        cfg)
                            continue
                        elif param.startswith('in:'):
                            idx = int(param[3:])
                            tids = [op.read[idx].id]
                            if tids[0] in self.grad_map.keys():
                                tids.append(self.grad_map[tids[0]])

                        for tid in tids:
                            self.data_config[tid].manual_pconfig(cfg)

    def Placeholder(self,
                    shape,
                    dtype=DataType.Float32,
                    is_label=False,
                    requires_grad=False,
                    name=''):
        out = Tensor(shape,
                     dtype=dtype,
                     requires_grad=requires_grad,
                     name=name if name else 'x')
        out = Input(out)
        self.datas[out.id] = out
        if self.train and requires_grad:
            grad = Tensor(shape,
                          dtype=dtype,
                          name=name + '.grad' if name else 'x.grad')
            grad = Gradient(grad, out)
            self.grad_map[out.id] = grad.id
            self.datas[grad.id] = grad
        self.inputs.append(out.id)
        return out

    def Linear(self, input, weight, bias=None, pconfig=None, name=''):
        # build forward tensors
        self._parameters.append(weight.id)
        inputs = (input, weight)
        if bias is not None:
            self._parameters.append(bias.id)
            inputs = (input, weight, bias)
        size = list(input.size())
        size[-1] = weight.size(0)
        out = Tensor(size, name='y')
        # build op
        op = Linear(inputs, (out, ), name=name)
        # add data nodes
        for data in inputs[1:]:
            self.datas[data.id] = data
        self.datas[out.id] = out
        # add op nodes
        self.ops[op.id] = op
        # add edges
        for i, data in enumerate(inputs):
            self.edges.add(Edge('r', op.id, i, data.id))
        self.edges.add(Edge('w', op.id, 0, out.id))
        # add backward
        if self.train:
            # build backward tensors
            if weight.grad is not None:
                dweight = weight.grad
            else:
                dweight = Tensor(weight.size(), dtype=weight.dtype, name='dw')
                dweight = Gradient(dweight, weight)
                self.grad_map[weight.id] = dweight.id
            if input.requires_grad:
                outputs = [self.datas[self.grad_map[input.id]], dweight]
            else:
                outputs = [dweight]
            if bias is not None:
                dbias = Tensor(bias.size(), dtype=bias.dtype, name='db')
                dbias = Gradient(dbias, bias)
                self.grad_map[bias.id] = dbias.id
                outputs.append(dbias)
            dout = Tensor(out.size(), dtype=out.dtype, name='dy')
            dout = Gradient(dout, out)
            self.grad_map[out.id] = dout.id
            # build backward op
            op_bw = LinearBW((dout, input, weight), outputs)
            self.fwop_map[op.id] = op_bw.id
            # add data nodes
            for data in outputs:
                self.datas[data.id] = data
            self.datas[dout.id] = dout
            # add op nodes
            self.ops[op_bw.id] = op_bw
            # add edges
            for i, data in enumerate((dout, input, weight)):
                self.edges.add(Edge('r', op_bw.id, i, data.id))
            for i, data in enumerate(outputs):
                self.edges.add(Edge('aw', op_bw.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return out

    def Matmul(self, in1, in2, pconfig=None, name=''):
        assert len(in1.size()) == len(in2.size()) or len(in2.size()) == 2
        size = list(in1.size())
        size[-1] = in2.size(-1)
        out = Tensor(size, name='y')
        # build op
        op = Matmul((in1, in2), (out, ), name=name)
        self.datas[out.id] = out
        # add op nodes
        self.ops[op.id] = op
        # add edges
        self.edges.add(Edge('r', op.id, 0, in1.id))
        self.edges.add(Edge('r', op.id, 1, in2.id))
        self.edges.add(Edge('w', op.id, 0, out.id))
        # add backward
        if self.train:
            # build backward tensors
            outputs = (self.datas[self.grad_map[in1.id]],
                       self.datas[self.grad_map[in2.id]])
            dout = Tensor(out.size(), dtype=out.dtype, name='dy')
            dout = Gradient(dout, out)
            self.grad_map[out.id] = dout.id
            # build backward op
            op_bw = MatmulBW((dout, in1, in2), outputs)
            self.fwop_map[op.id] = op_bw.id
            self.datas[dout.id] = dout
            # add op nodes
            self.ops[op_bw.id] = op_bw
            # add edges
            self.edges.add(Edge('r', op_bw.id, 0, dout.id))
            self.edges.add(Edge('r', op_bw.id, 1, in1.id))
            self.edges.add(Edge('r', op_bw.id, 2, in2.id))
            for i, data in enumerate(outputs):
                self.edges.add(Edge('aw', op_bw.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return out

    def Conv2d(self,
               input,
               weight,
               bias=None,
               stride=1,
               padding=0,
               pconfig=None,
               name=''):
        attr = {'stride': stride, 'padding': padding}
        kernel_size = weight.size()[2:]
        stride = _pair(stride)
        padding = _pair(padding)
        self._parameters.append(weight.id)
        inputs = (input, weight)
        if bias is not None:
            self._parameters.append(bias.id)
            inputs = (input, weight, bias)
        size = list(input.size())
        assert size[1] == input.size(1)
        size[1] = weight.size(0)
        for i in range(2):
            size[i + 2] = (size[i + 2] + 2 * padding[i] -
                           kernel_size[i]) // stride[i] + 1
        out = Tensor(size, name='y')
        # build op
        op = Conv2d(inputs, (out, ), attr=attr, name=name)
        # add data nodes
        for data in inputs[1:]:
            self.datas[data.id] = data
        self.datas[out.id] = out
        # add op nodes
        self.ops[op.id] = op
        # add edges
        for i, data in enumerate(inputs):
            self.edges.add(Edge('r', op.id, i, data.id))
        self.edges.add(Edge('w', op.id, 0, out.id))
        # add backward
        if self.train:
            # build backward tensors
            dweight = Tensor(weight.size(), dtype=weight.dtype, name='dw')
            dweight = Gradient(dweight, weight)
            self.grad_map[weight.id] = dweight.id
            if input.requires_grad:
                outputs = [self.datas[self.grad_map[input.id]], dweight]
            else:
                outputs = [dweight]
            if bias is not None:
                dbias = Tensor(bias.size(), dtype=bias.dtype, name='db')
                dbias = Gradient(dbias, bias)
                self.grad_map[bias.id] = dbias.id
                outputs.append(dbias)
            dout = Tensor(out.size(), dtype=out.dtype, name='dy')
            dout = Gradient(dout, out)
            self.grad_map[out.id] = dout.id
            # build backward op
            op_bw = Conv2dBW((dout, input, weight), outputs, attr=attr)
            self.fwop_map[op.id] = op_bw.id
            # add data nodes
            for data in outputs:
                self.datas[data.id] = data
            self.datas[dout.id] = dout
            # add op nodes
            self.ops[op_bw.id] = op_bw
            # add edges
            for i, data in enumerate((dout, input, weight)):
                self.edges.add(Edge('r', op_bw.id, i, data.id))
            for i, data in enumerate(outputs):
                self.edges.add(Edge('aw', op_bw.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return out

    def BatchNorm2d(self,
                    input,
                    weight=None,
                    bias=None,
                    pconfig=None,
                    name=''):
        # build forward tensors
        self._parameters.append(weight.id)
        self._parameters.append(bias.id)
        inputs = (input, weight, bias)
        out = Tensor(input.size(), name='y')
        # build op
        op = BatchNorm2d(inputs, (out, ), name=name)
        # add data nodes
        for data in inputs[1:]:
            self.datas[data.id] = data
        self.datas[out.id] = out
        # add op nodes
        self.ops[op.id] = op
        # add edges
        for i, data in enumerate(inputs):
            self.edges.add(Edge('r', op.id, i, data.id))
        self.edges.add(Edge('w', op.id, 0, out.id))
        # add backward
        if self.train:
            # build backward tensors
            dweight = Tensor(weight.size(), dtype=weight.dtype, name='dw')
            dweight = Gradient(dweight, weight)
            self.grad_map[weight.id] = dweight.id
            dbias = Tensor(bias.size(), dtype=bias.dtype, name='db')
            dbias = Gradient(dbias, bias)
            self.grad_map[bias.id] = dbias.id
            outputs = (self.datas[self.grad_map[input.id]], dweight, dbias)
            dout = Tensor(out.size(), dtype=out.dtype, name='dy')
            dout = Gradient(dout, out)
            self.grad_map[out.id] = dout.id
            # build backward op
            op_bw = BatchNorm2dBW((dout, input, weight), outputs)
            self.fwop_map[op.id] = op_bw.id
            # add data nodes
            for data in outputs[1:]:
                self.datas[data.id] = data
            self.datas[dout.id] = dout
            # add op nodes
            self.ops[op_bw.id] = op_bw
            # add edges
            for i, data in enumerate((dout, input, weight)):
                self.edges.add(Edge('r', op_bw.id, i, data.id))
            for i, data in enumerate(outputs):
                self.edges.add(Edge('aw', op_bw.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return out

    def LayerNorm(self, input, weight=None, bias=None, pconfig=None, name=''):
        # build forward tensors
        self._parameters.append(weight.id)
        self._parameters.append(bias.id)
        inputs = (input, weight, bias)
        out = Tensor(input.size(), name='y')
        # build op
        op = LayerNorm(inputs, (out, ), weight.size(0), name=name)
        # add data nodes
        for data in inputs[1:]:
            self.datas[data.id] = data
        self.datas[out.id] = out
        # add op nodes
        self.ops[op.id] = op
        # add edges
        for i, data in enumerate(inputs):
            self.edges.add(Edge('r', op.id, i, data.id))
        self.edges.add(Edge('w', op.id, 0, out.id))
        # add backward
        if self.train:
            # build backward tensors
            dweight = Tensor(weight.size(), dtype=weight.dtype, name='dw')
            dweight = Gradient(dweight, weight)
            self.grad_map[weight.id] = dweight.id
            dbias = Tensor(bias.size(), dtype=bias.dtype, name='db')
            dbias = Gradient(dbias, bias)
            self.grad_map[bias.id] = dbias.id
            outputs = (self.datas[self.grad_map[input.id]], dweight, dbias)
            dout = Tensor(out.size(), dtype=out.dtype, name='dy')
            dout = Gradient(dout, out)
            self.grad_map[out.id] = dout.id
            # build backward op
            op_bw = LayerNormBW((dout, input, weight), outputs, weight.size(0))
            self.fwop_map[op.id] = op_bw.id
            # add data nodes
            for data in outputs[1:]:
                self.datas[data.id] = data
            self.datas[dout.id] = dout
            # add op nodes
            self.ops[op_bw.id] = op_bw
            # add edges
            for i, data in enumerate((dout, input, weight)):
                self.edges.add(Edge('r', op_bw.id, i, data.id))
            for i, data in enumerate(outputs):
                self.edges.add(Edge('aw', op_bw.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return out

    def ReLU(self, input, pconfig=None, name=''):
        # build forward tensors
        out = Tensor(input.size(), name='y')
        # build op
        op = ReLU((input, ), (out, ), name=name)
        self.datas[out.id] = out
        # add op nodes
        self.ops[op.id] = op
        # add edges
        self.edges.add(Edge('r', op.id, 0, input.id))
        self.edges.add(Edge('w', op.id, 0, out.id))
        # add backward
        if self.train:
            # build backward tensors
            outputs = (self.datas[self.grad_map[input.id]], )
            dout = Tensor(out.size(), dtype=out.dtype, name='dy')
            dout = Gradient(dout, out)
            self.grad_map[out.id] = dout.id
            # build backward op
            # op_bw = ReLUBW((dout, input), outputs)
            op_bw = ReLUBW((dout, out), outputs)
            self.fwop_map[op.id] = op_bw.id
            self.datas[dout.id] = dout
            # add op nodes
            self.ops[op_bw.id] = op_bw
            # add edges
            self.edges.add(Edge('r', op_bw.id, 0, dout.id))
            self.edges.add(Edge('r', op_bw.id, 1, out.id))
            for i, data in enumerate(outputs):
                self.edges.add(Edge('aw', op_bw.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return out

    def Activation(self, input, act_type=None, pconfig=None, name=''):
        # build forward tensors
        out = Tensor(input.size(), name='y')
        # build op
        op = Activation((input, ), (out, ), act_type, name=name)
        self.datas[out.id] = out
        # add op nodes
        self.ops[op.id] = op
        # add edges
        self.edges.add(Edge('r', op.id, 0, input.id))
        self.edges.add(Edge('w', op.id, 0, out.id))
        # add backward
        if self.train:
            # build backward tensors
            outputs = (self.datas[self.grad_map[input.id]], )
            dout = Tensor(out.size(), dtype=out.dtype, name='dy')
            dout = Gradient(dout, out)
            self.grad_map[out.id] = dout.id
            # build backward op
            op_bw = ActivationBW((dout, out), outputs, act_type)
            self.fwop_map[op.id] = op_bw.id
            self.datas[dout.id] = dout
            # set associated buffer
            if act_type == 'gelu':
                out.set_associated_buffer(out.dtype, 4)
            # add op nodes
            self.ops[op_bw.id] = op_bw
            # add edges
            self.edges.add(Edge('r', op_bw.id, 0, dout.id))
            self.edges.add(Edge('r', op_bw.id, 1, out.id))
            for i, data in enumerate(outputs):
                self.edges.add(Edge('aw', op_bw.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return out

    def Dropout(self, input, p=0.5, pconfig=None, name=''):
        attr = {'p': p}
        out = Tensor(input.size(), name='y')
        # build op
        op = Dropout((input, ), (out, ), attr=attr, name=name)
        self.datas[out.id] = out
        # add op nodes
        self.ops[op.id] = op
        # add edges
        self.edges.add(Edge('r', op.id, 0, input.id))
        self.edges.add(Edge('w', op.id, 0, out.id))
        # add backward
        if self.train:
            # build backward tensors
            outputs = (self.datas[self.grad_map[input.id]], )
            dout = Tensor(out.size(), dtype=out.dtype, name='dy')
            dout = Gradient(dout, out)
            self.grad_map[out.id] = dout.id
            # build backward op
            op_bw = DropoutBW((dout, ), outputs, attr=attr)
            self.fwop_map[op.id] = op_bw.id
            self.datas[dout.id] = dout
            # add op nodes
            self.ops[op_bw.id] = op_bw
            # add edges
            self.edges.add(Edge('r', op_bw.id, 0, dout.id))
            for i, data in enumerate(outputs):
                self.edges.add(Edge('aw', op_bw.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return out

    def Pool2d(self,
               input,
               kernel_size,
               stride=None,
               padding=0,
               mode='max',
               pconfig=None,
               name=''):
        attr = {
            'mode': mode,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding
        }
        kernel_size = _pair(kernel_size)
        stride = kernel_size if stride is None else _pair(stride)
        padding = _pair(padding)
        size = list(input.size())
        for i in range(2):
            size[i + 2] = (size[i + 2] + 2 * padding[i] -
                           kernel_size[i]) // stride[i] + 1
        out = Tensor(size, name='y')
        # build op
        op = Pool2d((input, ), (out, ), attr=attr, name=name)
        self.datas[out.id] = out
        # add op nodes
        self.ops[op.id] = op
        # add edges
        self.edges.add(Edge('r', op.id, 0, input.id))
        self.edges.add(Edge('w', op.id, 0, out.id))
        # add backward
        if self.train:
            # build backward tensors
            outputs = (self.datas[self.grad_map[input.id]], )
            dout = Tensor(out.size(), dtype=out.dtype, name='dy')
            dout = Gradient(dout, out)
            self.grad_map[out.id] = dout.id
            # build backward op
            op_bw = Pool2dBW((dout, ), outputs, attr=attr)
            self.fwop_map[op.id] = op_bw.id
            self.datas[dout.id] = dout
            # add op nodes
            self.ops[op_bw.id] = op_bw
            # add edges
            self.edges.add(Edge('r', op_bw.id, 0, dout.id))
            for i, data in enumerate(outputs):
                self.edges.add(Edge('aw', op_bw.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return out

    def AdaptivePool2d(self,
                       input,
                       output_size,
                       mode='avg',
                       pconfig=None,
                       name=''):
        attr = {'mode': mode, 'output_size': output_size}
        output_size = _pair(output_size)
        size = [input.size(0), input.size(1)] + list(output_size)
        out = Tensor(size, name='y')
        # build op
        op = AdaptivePool2d((input, ), (out, ), attr=attr, name=name)
        self.datas[out.id] = out
        # add op nodes
        self.ops[op.id] = op
        # add edges
        self.edges.add(Edge('r', op.id, 0, input.id))
        self.edges.add(Edge('w', op.id, 0, out.id))
        # add backward
        if self.train:
            # build backward tensors
            outputs = (self.datas[self.grad_map[input.id]], )
            dout = Tensor(out.size(), dtype=out.dtype, name='dy')
            dout = Gradient(dout, out)
            self.grad_map[out.id] = dout.id
            # build backward op
            op_bw = AdaptivePool2dBW((dout, ), outputs, attr=attr)
            self.fwop_map[op.id] = op_bw.id
            self.datas[dout.id] = dout
            # add op nodes
            self.ops[op_bw.id] = op_bw
            # add edges
            self.edges.add(Edge('r', op_bw.id, 0, dout.id))
            for i, data in enumerate(outputs):
                self.edges.add(Edge('aw', op_bw.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return out

    def Reshape(self, input, *size, pconfig=None, name=''):
        if isinstance(size[0], (tuple, list)):
            size = size[0]
        if -1 in size:
            hide = np.prod(input.size()) // np.abs(np.prod(size))
            size = [l if l != -1 else hide for l in size]
        out = Tensor(size, name='y')
        # build op
        op = Reshape((input, ), (out, ), name=name)
        self.datas[out.id] = out
        # add op nodes
        self.ops[op.id] = op
        # add edges
        self.edges.add(Edge('r', op.id, 0, input.id))
        self.edges.add(Edge('w', op.id, 0, out.id))
        if self.train:
            # build backward tensors
            outputs = (self.datas[self.grad_map[input.id]], )
            dout = Tensor(out.size(), dtype=out.dtype, name='dy')
            dout = Gradient(dout, out)
            self.grad_map[out.id] = dout.id
            # build backward op
            op_bw = Reshape((dout, ), outputs, bw=True)
            self.fwop_map[op.id] = op_bw.id
            self.datas[dout.id] = dout
            # add op nodes
            self.ops[op_bw.id] = op_bw
            # add edges
            self.edges.add(Edge('r', op_bw.id, 0, dout.id))
            for i, data in enumerate(outputs):
                self.edges.add(Edge('aw', op_bw.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return out

    def Permute(self, input, *perm, pconfig=None, name=''):
        if len(perm) == 1 and isinstance(perm[0], (list, tuple)):
            perm = perm[0]
        size = list(input.size())
        for i, p in enumerate(perm):
            size[i] = input.size(p)
        out = Tensor(size, name='y')
        # build op
        op = Permute((input, ), (out, ), perm, name=name)
        self.datas[out.id] = out
        # add op nodes
        self.ops[op.id] = op
        # add edges
        self.edges.add(Edge('r', op.id, 0, input.id))
        self.edges.add(Edge('w', op.id, 0, out.id))
        if self.train:
            # build backward tensors
            outputs = (self.datas[self.grad_map[input.id]], )
            dout = Tensor(out.size(), dtype=out.dtype, name='dy')
            dout = Gradient(dout, out)
            self.grad_map[out.id] = dout.id
            # build backward op
            op_bw = Permute((dout, ), outputs, perm, bw=True)
            self.fwop_map[op.id] = op_bw.id
            self.datas[dout.id] = dout
            # add op nodes
            self.ops[op_bw.id] = op_bw
            # add edges
            self.edges.add(Edge('r', op_bw.id, 0, dout.id))
            for i, data in enumerate(outputs):
                self.edges.add(Edge('aw', op_bw.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return out

    def Split(self, input, size_or_sections, dim=0, pconfig=None, name=''):
        assert isinstance(dim, int)
        attr = {'size_or_sections': size_or_sections, 'dim': dim}
        if isinstance(size_or_sections, int):
            size = size_or_sections
            sections = [size] * (input.size(dim) // size)
            if input.size(dim) % size != 0:
                sections.append(input.size(dim) % size)
        else:
            sections = size_or_sections
        assert isinstance(sections, (tuple, list)) and len(sections) > 0
        outs = []
        for i, sec in enumerate(sections):
            size = list(input.size())
            size[dim] = sec
            out = Tensor(size, name='y' + str(i))
            outs.append(out)
        # build op
        op = Split((input, ), outs, dim, attr=attr, name=name)
        for out in outs:
            self.datas[out.id] = out
        # add op nodes
        self.ops[op.id] = op
        # add edges
        self.edges.add(Edge('r', op.id, 0, input.id))
        for i, out in enumerate(outs):
            self.edges.add(Edge('w', op.id, i, out.id))
        if self.train:
            # build backward tensors
            outputs = (self.datas[self.grad_map[input.id]], )
            douts = []
            for i, out in enumerate(outs):
                dout = Tensor(out.size(), dtype=out.dtype, name='dy' + str(i))
                dout = Gradient(dout, out)
                self.grad_map[out.id] = dout.id
                douts.append(dout)
            # build backward op
            op_bw = Concat(douts, outputs, dim)
            self.fwop_map[op.id] = op_bw.id
            for dout in douts:
                self.datas[dout.id] = dout
            # add op nodes
            self.ops[op_bw.id] = op_bw
            # add edges
            for i, dout in enumerate(douts):
                self.edges.add(Edge('r', op_bw.id, i, dout.id))
            for i, data in enumerate(outputs):
                self.edges.add(Edge('aw', op_bw.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return tuple(outs)

    def Concat(self, inputs, dim=0, pconfig=None, name=''):
        assert isinstance(inputs, (tuple, list))
        assert isinstance(dim, int)
        size = list(inputs[0].size())
        size[dim] = sum([t.size(dim) for t in inputs])
        out = Tensor(size, name='y')
        # build op
        op = Concat(inputs, (out, ), dim, name=name)
        self.datas[out.id] = out
        # add op nodes
        self.ops[op.id] = op
        # add edges
        for i, input in enumerate(inputs):
            self.edges.add(Edge('r', op.id, i, input.id))
        self.edges.add(Edge('w', op.id, 0, out.id))
        if self.train:
            # build backward tensors
            outputs = []
            for input in inputs:
                outputs.append(self.datas[self.grad_map[input.id]])
            dout = Tensor(out.size(), dtype=out.dtype, name='dy')
            dout = Gradient(dout, out)
            self.grad_map[out.id] = dout.id
            # build backward op
            sections = []
            for input in inputs:
                sections.append(input.size(dim))
            attr = {'size_or_sections': sections, 'dim': dim}
            op_bw = Split((dout, ), outputs, dim, attr=attr)
            self.fwop_map[op.id] = op_bw.id
            self.datas[dout.id] = dout
            # add op nodes
            self.ops[op_bw.id] = op_bw
            # add edges
            self.edges.add(Edge('r', op_bw.id, 0, dout.id))
            for i, data in enumerate(outputs):
                self.edges.add(Edge('aw', op_bw.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return out

    def Slice(self, input, nelements, pconfig=None, name=''):
        size = list(input.size())
        size[-1] = nelements
        out = Tensor(size, name='y')
        op = SliceFW((input, ), (out, ), nelements, name=name)

        self.datas[out.id] = out
        self.ops[op.id] = op
        # add edges
        self.edges.add(Edge('r', op.id, 0, input.id))
        self.edges.add(Edge('w', op.id, 0, out.id))
        if self.train:
            # build backward tensors
            outputs = (self.datas[self.grad_map[input.id]], )
            dout = Tensor(out.size(), dtype=out.dtype, name='dy')
            dout = Gradient(dout, out)
            self.grad_map[out.id] = dout.id
            # build backward op
            op_bw = SliceBW((dout, ), outputs, nelements)
            self.fwop_map[op.id] = op_bw.id
            self.datas[dout.id] = dout
            # add op nodes
            self.ops[op_bw.id] = op_bw
            # add edges
            self.edges.add(Edge('r', op_bw.id, 0, dout.id))
            for i, data in enumerate(outputs):
                self.edges.add(Edge('aw', op_bw.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return out


    def Softmax(self, input, dim=0, pconfig=None, name=''):
        # build forward tensors
        out = Tensor(input.size(), name='y')
        # build op
        op = Softmax((input, ), (out, ), dim=dim, name=name)
        self.datas[out.id] = out
        # add op nodes
        self.ops[op.id] = op
        # add edges
        self.edges.add(Edge('r', op.id, 0, input.id))
        self.edges.add(Edge('w', op.id, 0, out.id))
        # add backward
        if self.train:
            # build backward tensors
            outputs = (self.datas[self.grad_map[input.id]], )
            dout = Tensor(out.size(), dtype=out.dtype, name='dy')
            dout = Gradient(dout, out)
            self.grad_map[out.id] = dout.id
            # build backward op
            op_bw = SoftmaxBW((dout, out), outputs, dim=dim)
            self.fwop_map[op.id] = op_bw.id
            self.datas[dout.id] = dout
            # set associated buffer
            out.set_associated_buffer(out.dtype, 1)
            # add op nodes
            self.ops[op_bw.id] = op_bw
            # add edges
            self.edges.add(Edge('r', op_bw.id, 0, dout.id))
            self.edges.add(Edge('r', op_bw.id, 1, out.id))
            for i, data in enumerate(outputs):
                self.edges.add(Edge('aw', op_bw.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return out

    def CrossEntropyLoss(self, input, target, loss_type=None, pconfig=None, name=''):
        # ignore loss in computation plan graph
        unreduced_loss = Tensor(input.size(0), name='loss')
        unreduced_loss = Output(unreduced_loss)
        op = CrossEntropy((input, target), (unreduced_loss, ), loss_type=loss_type, name=name)
        self.datas[unreduced_loss.id] = unreduced_loss
        self.ops[op.id] = op
        for i, data in enumerate((input, target)):
            self.edges.add(Edge('r', op.id, i, data.id))
        self.edges.add(Edge('w', op.id, 0, unreduced_loss.id))
        if self.train:
            dout = Tensor(unreduced_loss.size(),
                          dtype=unreduced_loss.dtype,
                          name='dloss')
            dout = Gradient(dout, unreduced_loss)
            self.grad_map[unreduced_loss.id] = dout.id
            op_bw = CrossEntropyBW((dout, target),
                                   (self.datas[self.grad_map[input.id]], ),
                                   loss_type=loss_type)
            self.fwop_map[op.id] = op_bw.id
            self.datas[dout.id] = dout
            self.ops[op_bw.id] = op_bw
            for i, data in enumerate((dout, target)):
                self.edges.add(Edge('r', op_bw.id, i, data.id))
            self.edges.add(Edge('aw', op_bw.id, 0, self.grad_map[input.id]))
            dout.control.append(unreduced_loss)
            # delete grad of target
            if target.id in self.grad_map.keys():
                del self.datas[self.grad_map[target.id]]
                del self.grad_map[target.id]
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return unreduced_loss

    def Elementwise(self, in1, in2, type: str, pconfig=None, name=''):
        # build forward tensors
        size = []
        for s1, s2 in zip(in1.size(), in2.size()):
            size.append(max(s1, s2))
        out = Tensor(size, name='y')
        # build op
        op = Elementwise((in1, in2), (out, ), type, name=name)
        self.datas[out.id] = out
        # add op nodes
        self.ops[op.id] = op
        # add edges
        self.edges.add(Edge('r', op.id, 0, in1.id))
        self.edges.add(Edge('r', op.id, 1, in2.id))
        self.edges.add(Edge('w', op.id, 0, out.id))
        # add backward
        if self.train:
            # build backward tensors
            if in2.requires_grad == False:
                outputs = (self.datas[self.grad_map[in1.id]], )
            else:
                outputs = (self.datas[self.grad_map[in1.id]],
                           self.datas[self.grad_map[in2.id]])
            dout = Tensor(out.size(), dtype=out.dtype, name='dy')
            dout = Gradient(dout, out)
            self.grad_map[out.id] = dout.id
            # build backward op
            input_shape = (in1.size(), in2.size())
            if type in ['mul', 'div', 'sqrt', 'attention_mask']:
                inputs = (dout, in1, in2)
            else:
                inputs = (dout, )
            op_bw = ElementwiseBW(inputs, outputs, type, input_shape)
            self.fwop_map[op.id] = op_bw.id
            self.datas[dout.id] = dout
            # add op nodes
            self.ops[op_bw.id] = op_bw
            # add edges
            for i, data in enumerate(inputs):
                self.edges.add(Edge('r', op_bw.id, i, data.id))
            for i, data in enumerate(outputs):
                self.edges.add(Edge('aw', op_bw.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return out

    def Embedding(self, input, weight, attr=None, pconfig=None, name=''):
        # build forward tensors
        self._parameters.append(weight.id)
        inputs = (input, weight)
        size = list(input.size())
        size.append(weight.size(1))
        out = Tensor(size, name='y_embd')
        # build op
        op = Embedding(inputs, (out, ), attr=attr, name=name)
        # add data nodes
        for data in inputs[1:]:
            self.datas[data.id] = data
        self.datas[out.id] = out
        # add op nodes
        self.ops[op.id] = op
        # add edges
        for i, data in enumerate(inputs):
            self.edges.add(Edge('r', op.id, i, data.id))
        self.edges.add(Edge('w', op.id, 0, out.id))
        # add backward
        if self.train:
            # build backward tensors
            dweight = Tensor(weight.size(), dtype=weight.dtype, name='dw')
            dweight = Gradient(dweight, weight)
            self.grad_map[weight.id] = dweight.id
            outputs = [dweight]
            dout = Tensor(out.size(), dtype=out.dtype, name='dy_embd')
            dout = Gradient(dout, out)
            self.grad_map[out.id] = dout.id
            # build backward op
            op_bw = EmbeddingBW((dout, input), outputs, attr=attr)
            self.fwop_map[op.id] = op_bw.id
            # add data nodes
            for data in outputs:
                self.datas[data.id] = data
            self.datas[dout.id] = dout
            # add op nodes
            self.ops[op_bw.id] = op_bw
            # add edges
            for i, data in enumerate((dout, input)):
                self.edges.add(Edge('r', op_bw.id, i, data.id))
            for i, data in enumerate(outputs):
                self.edges.add(Edge('aw', op_bw.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig((op, op_bw) if self.train else (op, ), pconfig)
        return out

    def Buffer(self, shape, dtype=DataType.Float32):
        out = Tensor(shape, dtype=dtype, name='buffer')
        out = Buffer(out)
        self.datas[out.id] = out
        return out

    def SGD(self, param, grad, buf=None, attr=None, pconfig=None):
        ins = (param, grad) if buf is None else (param, grad, buf)
        op = SGD(ins, (), attr=attr)
        # add next version parameter
        op.write = (param, )
        op.outs = (param.size(), )
        param.set_optimizer(op)
        self.ops[op.id] = op
        for i, data in enumerate(ins):
            self.edges.add(Edge('rw', op.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig(op, pconfig)
        return op

    def Adam(self, param, grad, exp_avg, exp_avg_sqr, attr=None, pconfig=None):
        ins = (param, grad, exp_avg, exp_avg_sqr)
        op = Adam(ins, (), attr=attr)
        self.ops[op.id] = op
        # add next version parameter
        op.write = (param, )
        op.outs = (param.size(), )
        param.set_optimizer(op)
        for i, data in enumerate(ins):
            self.edges.add(Edge('rw', op.id, i, data.id))
        if pconfig is not None:
            self.set_op_pconfig(op, pconfig)
        else:
            if hasattr(self,
                       'data_config') and grad.id in self.data_config.keys():
                gcfg = self.data_config[grad.id]
                if gcfg.manual:
                    for data in [exp_avg, exp_avg_sqr]:
                        if data.id not in self.data_config.keys():
                            self.data_config[data.id] = TensorConfig(
                                data, self.dev_topo.ndevs)
                        self.data_config[data.id].set_config(gcfg.get_config())
        return op

    def parameters(self):
        params = [self.datas[i] for i in self._parameters]
        return params

    def to_graphviz(self, file='log/graph_tree'):
        graph = Digraph()
        for k, v in self.ops.items():
            graph.node('c' + str(k),
                       str(k) + ':' + enum_to_str(OpType, v.type))
        for k, v in self.datas.items():
            graph.node('d' + str(k), v.name + ':' + str(k), shape='box')
        for e in self.edges:
            if e.privilege in ['r', 'rw']:
                graph.edge('d' + str(e.data), 'c' + str(e.op))
            else:
                graph.edge('c' + str(e.op), 'd' + str(e.data))
        with open(f'{file}.txt', 'w') as f:
            f.write(graph.source)
