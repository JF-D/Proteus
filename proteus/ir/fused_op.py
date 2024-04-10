from collections import defaultdict

from toposort import toposort_flatten
from proteus import OpType
from proteus.simulator.cost_model import CostModel
from .device import Device
from .graph import ProteusModel
from .node import Op, Edge
from .op_helper import IterSpace


class FusedOp(Op):
    def __init__(self, ins, outs, intra_ops, intra_edges, anchor):
        super().__init__(ins, outs)
        self.type = OpType.Fused
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

        self.anchor = anchor
        self.ops = intra_ops
        self.edges = intra_edges

        self.topo_order = self.get_topo_order()

        self.reset_aux_vars()

    def get_iter_space(self):
        self.iter_space = self.ops[self.anchor].get_iter_space()
        return self.iter_space

    def measure_cost(self, dev: Device, flops=False):
        cost = 0
        for _, op in self.ops.items():
            cost += op.measure_cost(dev, flops=flops)
        if flops:
            return cost
        self.cost = cost
        return self.cost

    def reset_aux_vars(self, parts=None):
        self.old_parts = parts
        self.infered_ins = [False] * len(self.ins)
        self.infered_outs = [False] * len(self.outs)
        self.cache_in_parts = {}
        self.cache_out_parts = {}

    def adj_edge(self, reverse=False):
        adj = defaultdict(lambda: defaultdict(list))
        for (prev_id, wid, next_id, rid) in self.edges:
            if reverse:
                adj[next_id][rid].append((prev_id, wid))
            else:
                adj[prev_id][wid].append((next_id, rid))
        return adj

    def get_topo_order(self):
        adj = defaultdict(set)
        for (prev_id, wid, next_id, rid) in self.edges:
            if prev_id == -1 or next_id == -1:
                continue
            adj[next_id].add(prev_id)
        return toposort_flatten(adj)

    def propagate_parts(self, parts):
        f_edge = self.adj_edge()
        r_edge = self.adj_edge(reverse=True)

        parts_dt = {self.anchor: parts}
        anchor_topo_id = self.topo_order.index(self.anchor)
        for op_id in self.topo_order[anchor_topo_id:]:
            for wid, next_ops in f_edge[op_id].items():
                oparts = self.ops[op_id].get_output_parts(wid, parts_dt[op_id])
                for (nop_id, rid) in next_ops:
                    if nop_id == -1:
                        continue
                    it_space = self.ops[nop_id].iter_space
                    if nop_id not in parts_dt:
                        parts_dt[nop_id] = [1] * len(it_space.iters)
                    for idx, p in zip(it_space.in_iters[rid], oparts):
                        parts_dt[nop_id][idx] = p
        for op_id in self.topo_order[:anchor_topo_id + 1][::-1]:
            for rid, prev_ops in r_edge[op_id].items():
                iparts = self.ops[op_id].get_input_parts(rid, parts_dt[op_id])
                for (pop_id, wid) in prev_ops:
                    if pop_id == -1:
                        continue
                    it_space = self.ops[pop_id].iter_space
                    if pop_id not in parts_dt:
                        parts_dt[pop_id] = [1] * len(it_space.iters)
                    for idx, p in zip(it_space.out_iters[wid], iparts):
                        parts_dt[pop_id][idx] = p

        for rid, prev_ops in r_edge[-1].items():
            prev_op, wid = prev_ops[0]
            self.cache_out_parts[rid] = self.ops[prev_op].get_output_parts(
                wid, parts_dt[prev_op])
        for wid, next_ops in f_edge[-1].items():
            next_op, rid = next_ops[0]
            self.cache_in_parts[wid] = self.ops[next_op].get_input_parts(
                rid, parts_dt[next_op])

    def get_input_parts(self, in_id, parts):
        if parts == self.old_parts and self.infered_ins[in_id]:
            return self.cache_in_parts[in_id]
        self.reset_aux_vars(parts.copy())
        self.propagate_parts(parts)
        return self.cache_in_parts[in_id]

    def get_output_parts(self, out_id, parts):
        if parts == self.old_parts and self.infered_outs[out_id]:
            return self.cache_out_parts[out_id]
        self.reset_aux_vars(parts.copy())
        self.propagate_parts(parts)
        return self.cache_out_parts[out_id]

    def __repr__(self):
        string = str(self.type) + '_' + str(self.id) + '/'
        return string + self.ops[self.anchor].__repr__()

    @staticmethod
    def create(graph: ProteusModel, group: list, anchor=None):
        assert anchor is None or len(anchor) <= 1
        e_data, e_op = graph.forward_edge_set_to_adj(forward=False,
                                                     reverse=False)
        re_data, re_op = graph.forward_edge_set_to_adj(forward=False,
                                                       reverse=True)

        # get in and out data ids
        in_ids, out_ids = set(), set()
        op_ins, op_outs = set(), set()
        for op_id in group:
            op_ins.update(re_op[op_id])
            op_outs.update(e_op[op_id])
        for (did, _) in op_ins:
            internal = True
            for (prev, _) in re_data[did]:
                if prev not in group:
                    internal = False
                    break
            if len(re_data[did]) == 0:
                internal = False
            if not internal:
                in_ids.add(did)
        for (did, _) in op_outs:
            internal = True
            for (nid, _) in e_data[did]:
                if nid not in group:
                    internal = False
                    break
            for (prev, _) in re_data[did]:
                if prev not in group:
                    internal = False
                    break
            if not internal:
                out_ids.add(did)
        in_ids = sorted(list(in_ids))
        out_ids = sorted(list(out_ids))

        # modify edges
        fuseop_id = Op.id
        intra_ops, datas, inter_edges, intra_edges = {}, {}, set(), set()
        for op_id in group:
            intra_ops[op_id] = graph.ops.pop(op_id)

            for (data_id, rid) in re_op[op_id]:
                if data_id in in_ids:
                    intra_edges.add((-1, in_ids.index(data_id), op_id, rid))
                    inter_edges.add(
                        Edge('r', fuseop_id, in_ids.index(data_id), data_id))
                    graph.edges.discard(('r', op_id, rid, data_id))
                    graph.edges.discard(('rw', op_id, rid, data_id))
                else:
                    for (prev, wid) in re_data[data_id]:
                        assert prev in group
                        intra_edges.add((prev, wid, op_id, rid))
                        graph.edges.discard(('r', op_id, rid, data_id))
                        graph.edges.discard(('rw', op_id, rid, data_id))
                        graph.edges.discard(('w', prev, wid, data_id))
                        graph.edges.discard(('aw', prev, wid, data_id))

            for (data_id, wid) in e_op[op_id]:
                if data_id in out_ids:
                    intra_edges.add((op_id, wid, -1, out_ids.index(data_id)))
                    pr = 'aw' if len(re_data[data_id]) == 1 else 'w'
                    inter_edges.add(
                        Edge(pr, fuseop_id, out_ids.index(data_id), data_id))
                else:
                    datas[data_id] = graph.datas.pop(data_id)
                graph.edges.discard(('w', op_id, wid, data_id))
                graph.edges.discard(('aw', op_id, wid, data_id))

                for (nop_id, rid) in e_data[data_id]:
                    if nop_id in group:
                        graph.edges.discard(('r', nop_id, rid, data_id))
                        graph.edges.discard(('rw', nop_id, rid, data_id))
                        intra_edges.add((op_id, wid, nop_id, rid))

        ins = [graph.datas[did] for did in in_ids]
        outs = [graph.datas[did] for did in out_ids]

        # set anchor
        if anchor is None or len(anchor) == 0:
            anchor = group[-1]
            for op_id in group:
                if intra_ops[op_id].type in [
                        OpType.Linear, OpType.LinearBW, OpType.Conv2d,
                        OpType.Conv2dBW, OpType.Matmul, OpType.MatmulBW
                ]:
                    anchor = op_id
                    break
        else:
            assert len(anchor) == 1
            anchor = anchor.pop()
        fuse_op = FusedOp(ins, outs, intra_ops, intra_edges, anchor)
        assert fuse_op.id == fuseop_id
        graph.ops[fuse_op.id] = fuse_op
        graph.edges.update(inter_edges)
        return fuse_op
