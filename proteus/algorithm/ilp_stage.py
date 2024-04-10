from collections import defaultdict
import math
import cvxpy as cp
import numpy as np
from proteus import IterType
from proteus.ir import ProteusModel, Gradient
from proteus.ir import Parameter
from proteus.ir.ops import Adam, SGD
from proteus.strategy import DeviceTopo

from .base import BaseAlgo
from .ilp_algo import CFG


class Vertex(object):
    def __init__(self, _id):
        super().__init__()
        self.id = _id
        self._prev = []
        self._next = []

    def add_prev_vertex(self, vid, idx):
        self._prev.append((vid, idx))

    def add_next_vertex(self, vid, idx):
        self._next.append((vid, idx))

    @property
    def vprev(self):
        return self._prev

    @property
    def vnext(self):
        return self._next


class DVertex(Vertex):
    def __init__(self, data):
        super().__init__(data.id)
        self.shape = data.size()

    def center_op(self, pair=False):
        if len(self._prev) != 0:
            return self._prev[0] if pair else self._prev[0][0]
        cid = [v[0] for v in self._next]
        if pair:
            idx = np.argmin(cid)
            return self._next[idx]
        return min(cid)

    def adj_op(self, pair=False):
        cid = [v[0] for v in self._next]
        if len(self._prev) != 0:
            return self._next if pair else cid
        idx = np.argmin(cid)
        ret = []
        for i, vp in enumerate(self._next):
            if i != idx:
                ret.append(vp if pair else vp[0])
        return ret


class CVertex(Vertex):
    def __init__(self, op):
        super().__init__(op.id)
        self.op = op
        self.iter_space = op.iter_space

    def infer_input(self, in_id, parts):
        return self.op.get_input_parts(in_id, parts)

    def infer_output(self, out_id, parts):
        p = self.op.get_output_parts(out_id, parts)
        red = (np.prod(p) != np.prod(parts))
        return p, red


bvar_n, intvar_n, cvar_n = 0, 0, 0


def MakeVar(shape=(), name=None, type='float'):
    global bvar_n, intvar_n, cvar_n
    if not isinstance(shape, (list, tuple)):
        shape = (shape, )

    if type == 'bool':
        bvar_n += np.prod(shape)
        return cp.Variable(shape, name=name, boolean=True)
    elif type == 'int':
        intvar_n += np.prod(shape)
        return cp.Variable(shape, name=name, integer=True)
    elif type == 'int+':
        intvar_n += np.prod(shape)
        return cp.Variable(shape, name=name, integer=True, nonneg=True)
    elif type == 'float':
        cvar_n += np.prod(shape)
        return cp.Variable(shape, name=name)
    elif type == 'float+':
        cvar_n += np.prod(shape)
        return cp.Variable(shape, name=name, nonneg=True)
    else:
        assert False, 'Unknown type: {}'.format(type)


class StageILP(BaseAlgo):
    def __init__(self, graph: ProteusModel, dev_topo: DeviceTopo, scope='DMP'):
        super().__init__(dev_topo, scope)
        self.graph = graph
        self.dev_topo = dev_topo

        self.cfg_oracle = CFG(self.dev_topo.ndevs)
        for op_id in self.graph.ops:
            self.cfg_oracle.set_op(op_id,
                                   self.graph.ops[op_id].iter_space.iters,
                                   self.graph.op_config[op_id])

        self.dvs = {}
        self.cvs = {}
        for idx, data in self.graph.datas.items():
            self.dvs[idx] = DVertex(data)
        for idx, op in self.graph.ops.items():
            self.cvs[idx] = CVertex(op)
        for (pr, op_id, idx, data_id) in self.graph.edges:
            if pr in ['r', 'rw']:
                self.cvs[op_id].add_prev_vertex(data_id, idx)
                self.dvs[data_id].add_next_vertex(op_id, idx)
            else:
                self.cvs[op_id].add_next_vertex(data_id, idx)
                self.dvs[data_id].add_prev_vertex(op_id, idx)

    def optimize(self, verbose=True):
        self.stage1_problem = self.build_stage1_problem()
        self.stage1_problem.solve(verbose=verbose,
                                  solver=cp.CPLEX,
                                  cplex_filename='log/model.lp')
        self.get_stage1_solution()

    def build_stage1_problem(self, beta=1e-6):
        n, N = len(self.cvs), self.dev_topo.ndevs
        # var info
        global bvar_n, intvar_n, cvar_n
        bvar_n, intvar_n, cvar_n = 0, 0, 0

        self.P = {}
        for op_id in self.cvs:
            var = MakeVar(self.NCFG(op_id), 'P_' + str(op_id), 'bool')
            self.P[op_id] = var

        self.vcost = MakeVar(n, 'OpCost', 'float+')
        self.ecost, self.grad = {}, {}
        for d_id, dv in self.dvs.items():
            nvars = len(dv.vnext) + int(len(dv.vprev) > 0) - 1
            if isinstance(self.graph.datas[d_id], Parameter):
                self.grad[d_id] = MakeVar(name='grad_' + str(d_id), type='float+')
            if nvars <= 0:
                continue
            self.ecost[d_id] = MakeVar(nvars, 'EdgeCost_' + str(d_id), 'float+')

        self.I = defaultdict(dict)
        for d_id, dv in self.dvs.items():
            nvars = len(dv.vnext) + int(len(dv.vprev) > 0) - 1
            if nvars <= 0:
                continue
            ncenter = self.NCFG(dv.center_op())
            for i, adj_op_id in enumerate(dv.adj_op()):
                shape = (ncenter, self.NCFG(adj_op_id))
                self.I[d_id][i] = MakeVar(shape, f'I_{d_id}_{i}', 'bool')

        self.grad_reduce_cost = MakeVar(name='GradReduceCost', type='float+')
        self.act_exchg_cost = MakeVar(name='ActExcgCost', type='float+')
        self.max_exe_grad = MakeVar(name='OverLappedCost', type='float+')

        # statistics
        print('Total {} vars: {} binary, {} integer, {} continuous.'.format(
            bvar_n + intvar_n + cvar_n, bvar_n, intvar_n, cvar_n))

        constrs = self.make_stage1_constrs()

        # total_ecost = []
        # for d_id in self.ecost:
        #     total_ecost.append(cp.sum(self.ecost[d_id]))
        # obj = cp.Minimize(cp.sum(self.vcost) + 1e-6 * cp.sum(total_ecost))
        obj = cp.Minimize(self.max_exe_grad + self.act_exchg_cost)

        stage1_prob = cp.Problem(obj, constrs)
        return stage1_prob

    def make_stage1_constrs(self, beta=8e-7):
        n, N = len(self.cvs), self.dev_topo.ndevs
        constrs, cnt = [], 0

        def CSTR(cstr):
            nonlocal cnt
            cnt += cstr.size
            constrs.append(cstr)

        # op partition
        for i in self.cvs:
            CSTR(cp.sum(self.P[i]) == 1)

        # op cost
        for k, i in enumerate(self.cvs):
            exes = [
                self.P[i][c] * self.VCOST(i, c) for c in range(self.NCFG(i))
            ]
            CSTR(cp.sum(exes) == self.vcost[k])

        # indicator
        for d_id in self.I:
            dv = self.dvs[d_id]
            src = dv.center_op()
            ncenter = self.NCFG(src)
            for i, dst in enumerate(dv.adj_op()):
                for c in range(ncenter):
                    for h in range(self.NCFG(dst)):
                        CSTR(self.I[d_id][i][c, h] >= self.P[src][c] +
                             self.P[dst][h] - 1)
                        CSTR(self.I[d_id][i][c, h] <= self.P[src][c])
                        CSTR(self.I[d_id][i][c, h] <= self.P[dst][h])

        # edge cost
        grads, acts = [], []
        for d_id, dv in self.dvs.items():
            if d_id not in self.ecost:
                continue
            ncenter = self.NCFG(dv.center_op())
            for i, dst in enumerate(dv.adj_op()):
                exes = []
                for c in range(ncenter):
                    for h in range(self.NCFG(dst)):
                        exes.append(self.I[d_id][i][c, h] *
                                    self.ECOST(d_id, i, c, h) * beta)
                CSTR(self.ecost[d_id][i] >= cp.sum(exes))
                acts.append(self.ecost[d_id][i])
        for d_id in self.grad:
            op_id = self.dvs[d_id].center_op()
            exes = []
            for c in range(self.NCFG(op_id)):
                exes.append(self.P[op_id][c] * self.GCOST(d_id, c) * beta)
            CSTR(self.grad[d_id] >= cp.sum(exes))
            grads.append(self.grad[d_id])
        CSTR(cp.sum(grads) <= self.grad_reduce_cost)
        CSTR(cp.sum(acts) <= self.act_exchg_cost)

        CSTR(self.max_exe_grad >= cp.sum(self.vcost))
        CSTR(self.max_exe_grad >= self.grad_reduce_cost)

        # CSTR(beta * cp.sum(grads) <= self.grad_reduce_cost)
        # CSTR(beta * cp.sum(acts) <= self.act_exchg_cost)
        CSTR(self.max_exe_grad >= cp.sum(self.vcost) + self.grad_reduce_cost)

        # statistics
        print('Total {} constraints.'.format(cnt))
        return constrs

    def get_stage1_solution(self, file='log/stage1.txt'):
        configs = {}
        for op_id, varp in self.P.items():
            configs[op_id] = np.argmax(varp.value)

        with open(file, 'w') as f:
            for op_id, op in self.graph.ops.items():
                f.write('{}: {}\n'.format(
                    op, self.cfg_oracle.config(op_id, configs[op_id])))
            for var in self.stage1_problem.variables():
                f.write('{}: {}\n'.format(var.name(), var.value))

    def NCFG(self, op_id):
        return self.cfg_oracle.ncfg(op_id)

    def DEG(self, op_id, c):
        return self.cfg_oracle.deg(op_id, c)

    def VCOST(self, op_id, c=None):
        exe_time = self.graph.ops[op_id].measure_cost(self.dev_topo.dev(0))
        if c == None:
            return exe_time
        parts = self.cfg_oracle.config(op_id, c)
        exe_time = exe_time / np.prod(parts)
        if exe_time < 1e-4:
            print('Warning: meet op cost less than 1e-4: {}'.format(exe_time))
        ranges = self.graph.op_config[op_id].ranges
        for p, r in zip(parts, ranges):
            if p >= r.end:
                exe_time = 1e4
        return exe_time

    def ECOST(self, d_id, idx, c, h):
        src, out_id = self.dvs[d_id].center_op(pair=True)
        dst, in_id = self.dvs[d_id].adj_op(pair=True)[idx]

        src_parts = self.cfg_oracle.config(src, c)
        dst_parts = self.cfg_oracle.config(dst, h)

        if len(self.dvs[d_id].vprev) == 0:
            src_parts = self.cvs[src].infer_input(out_id, src_parts)
            need_red = False
        else:
            src_parts, need_red = self.cvs[src].infer_output(out_id, src_parts)
        dst_parts = self.cvs[dst].infer_input(in_id, dst_parts)

        shape = self.dvs[d_id].shape
        nelements = 0
        if need_red:
            nelements += np.prod(shape) // np.prod(src_parts)
        if src_parts != dst_parts:
            acc = 1
            for i, (p1, p2) in enumerate(zip(src_parts, dst_parts)):
                acc *= shape[i] // (max(p1, p2) // (min(p1, p2)))
            nelements += acc
        return nelements

    def GCOST(self, d_id, c):
        src, in_id = self.dvs[d_id].center_op(pair=True)
        src_parts = self.cfg_oracle.config(src, c)
        if src_parts[0] == 1:
            return 0

        src_parts = self.cvs[src].infer_input(in_id, src_parts)
        shape = self.dvs[d_id].shape
        nelements = np.prod(shape) / np.prod(src_parts)
        return nelements

    def IsGradReduce(self, d_id, idx):
        if not isinstance(self.graph.datas[d_id], Gradient):
            return False
        dst = self.dvs[d_id].adj_op()[idx]
        if isinstance(self.graph.ops[dst], (SGD, Adam)):
            return True
        return False
