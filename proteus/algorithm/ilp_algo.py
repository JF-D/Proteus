from collections import defaultdict
import math
import cvxpy as cp
import numpy as np
from numpy.core.numeric import Inf
from proteus import IterType
from proteus.ir import ProteusModel
from proteus.strategy import DeviceTopo

from .base import BaseAlgo

INF = 1e4


class CFG:
    def __init__(self, ndevs):
        self.cache = defaultdict(dict)
        self.mesh_2d = []
        self.mesh_1d = []
        for i in range(1, int(math.log2(ndevs)) + 1):
            self.mesh_1d.append(2**i)
            for j in range(1, int(math.log2(ndevs)) + 1):
                if (2**i) * (2**j) <= ndevs:
                    self.mesh_2d.append((2**i, 2**j))

        self.op_cfg = {}
        self._ncfg = {}
        self.niters = {}

    def set_op(self, op_id, iters, config):
        self.op_cfg[op_id] = []
        for i, (itype, irange) in enumerate(zip(iters, config.ranges)):
            if itype != IterType.OPAQUE:
                self.op_cfg[op_id].append((i, irange.end - 1))
            if i == 0:
                assert itype == IterType.BATCH

        self._ncfg[op_id] = {}
        self._ncfg[op_id][1] = len(self.op_cfg[op_id]) * len(self.mesh_1d)
        self._ncfg[op_id][2] = (len(self.op_cfg[op_id]) - 1) * len(self.mesh_2d)
        self._ncfg[op_id][0] = 1 + self._ncfg[op_id][1] + self._ncfg[op_id][2]
        self.niters[op_id] = len(iters)

    def ncfg(self, op_id):
        return self._ncfg[op_id][0]

    def deg(self, op_id, c):
        if c == 0:
            return 1
        c = c - 1
        if c // len(self.mesh_1d) < len(self.op_cfg[op_id]):
            mesh_id = c % len(self.mesh_1d)
            return self.mesh_1d[mesh_id]
        mesh_id = (c - self._ncfg[op_id][1]) % len(self.mesh_2d)
        product = lambda c: c[0] * c[1]
        return product(self.mesh_2d[mesh_id])

    def config(self, op_id, c):
        if c not in self.cache[op_id]:
            cfg = [1] * self.niters[op_id]
            if c == 0:
                pass
            elif (c - 1) // len(self.mesh_1d) < len(self.op_cfg[op_id]):
                mesh_id = (c - 1) % len(self.mesh_1d)
                dim_id = (c - 1) // len(self.mesh_1d)
                cfg[self.op_cfg[op_id][dim_id][0]] = self.mesh_1d[mesh_id]
            else:
                mesh_id = (c - 1 - self._ncfg[op_id][1]) % len(self.mesh_2d)
                dim_id = 1 + (c - 1 - self._ncfg[op_id][1]) // len(self.mesh_2d)
                cfg[0] = self.mesh_2d[mesh_id][0]
                cfg[self.op_cfg[op_id][dim_id][0]] = self.mesh_2d[mesh_id][1]
            self.cache[op_id][c] = cfg
        return self.cache[op_id][c]


class ILPAlgo(BaseAlgo):
    def __init__(self, graph: ProteusModel, dev_topo: DeviceTopo, scope='DM'):
        super().__init__(dev_topo, scope)
        self.graph = graph
        self.dev_topo = dev_topo
        self.edges, self.vpair = self.graph.make_op_graph()

        self.cfg_oracle = CFG(self.dev_topo.ndevs)
        for op_id in self.graph.ops:
            self.cfg_oracle.set_op(op_id,
                                   self.graph.ops[op_id].iter_space.iters,
                                   self.graph.op_config[op_id])

        self.build_problem()

    def build_problem(self):
        n, N = len(self.graph.ops), self.dev_topo.ndevs
        S = N
        pairs = [(op_i, op_j) for op_i in range(n) for op_j in range(n)
                 if op_i < op_j]
        spair = [(si, sj) for si in range(S) for sj in range(S) if si < sj]

        # var info
        bvar, intvar, cvar = 0, 0, 0

        self.P = []
        for op_id in range(n):
            var = cp.Variable(self.NCFG(op_id),
                              name='P_' + str(op_id),
                              boolean=True)
            self.P.append(var)
            bvar += var.size

        self.deg = cp.Variable(n, name='deg', integer=True)
        self.M = cp.Variable((n, N), name='M', boolean=True)
        self.W = cp.Variable((N, S), name='W', boolean=True)
        intvar += self.deg.size
        bvar += self.M.size + self.W.size

        # self.frac = cp.Variable(n, name='frac', nonneg=True)
        # cvar += self.frac.size
        self.time = cp.Variable(n, name='time', nonneg=True)
        cvar += self.time.size

        self.I = []
        for op_i, op_j in self.vpair.keys():
            var = cp.Variable((self.NCFG(op_i), self.NCFG(op_j)),
                              name=f'I_{op_i}_{op_j}',
                              boolean=True)
            self.I.append(var)
            bvar += var.size
        self.cost = cp.Variable(len(self.vpair), name='cost', nonneg=True)
        cvar += self.cost.size

        self.SO = cp.Variable(n, name='SO', nonneg=True)
        self.EO = cp.Variable(n, name='EO', nonneg=True)
        self.SE = cp.Variable(len(self.vpair), name='SE', nonneg=True)
        self.EE = cp.Variable(len(self.vpair), name='EE', nonneg=True)
        self.SD = cp.Variable(N, name='SD', nonneg=True)
        self.ED = cp.Variable(N, name='ED', nonneg=True)
        self.SS = cp.Variable(S, name='SS', nonneg=True)
        self.ES = cp.Variable(S, name='ES', nonneg=True)
        self.O = cp.Variable(name='O', nonneg=True)
        cvar += 2 * (n + len(self.vpair) + N + S) + 1

        self.U = cp.Variable(len(spair), name='U', boolean=True)
        bvar += self.U.size

        constrs = self.make_constraints()
        objective = cp.Minimize(self.O)
        self.problem = cp.Problem(objective, constrs)

        print('Total {} vars: {} binary, {} integer, {} continuous'.format(
            bvar + intvar + cvar, bvar, intvar, cvar))

    def make_constraints(self):
        constrs = []
        n, N = len(self.graph.ops), self.dev_topo.ndevs
        S = N
        pairs = [(op_i, op_j) for op_i in range(n) for op_j in range(n)
                 if op_i < op_j]
        spair = [(si, sj) for si in range(S) for sj in range(S) if si < sj]

        ncstrs = 0
        def ADD(cstr):
            nonlocal ncstrs
            ncstrs += cstr.size
            constrs.append(cstr)

        # 1> partition and map
        # op i use parallel config c
        for i in range(n):
            ADD(cp.sum(self.P[i]) == 1)
        # op i parallel degree
        for i in range(n):
            degs = [self.P[i][c] * self.DEG(i, c) for c in range(self.NCFG(i))]
            ADD(cp.sum(degs) == self.deg[i])
        # map op i to device j
        for i in range(n):
            ADD(cp.sum(self.M[i]) == self.deg[i])
        # map device j to stage s
        for j in range(N):
            ADD(cp.sum(self.W[j]) == 1)

        # 2> cost model
        # # fraction of total cost
        # for i in range(n):
        #     fractions = [N / self.DEG(i, c) for c in range(self.NCFG(i))]
        #     ADD(cp.sum(cp.multiply(self.P[i], fractions)) == self.frac[i] * N)
        # exec time of op i
        for i in range(n):
            exes = [self.P[i][c] * self.COST(i, c) for c in range(self.NCFG(i))]
            ADD(cp.sum(exes) == self.time[i])
            # ADD(self.frac[i] * self.COST(i) == self.time[i])
        # indicator of op i use config c and op j use config h
        for k, (opi, opj) in enumerate(self.vpair.keys()):
            for c in range(self.NCFG(opi)):
                for h in range(self.NCFG(opj)):
                    ADD(self.I[k][c, h] >= self.P[opi][c] + self.P[opj][h] - 1)
                    ADD(self.I[k][c, h] <= self.P[opi][c])
                    ADD(self.I[k][c, h] <= self.P[opj][h])
        # edge cost
        for k, (op_i, op_j) in enumerate(self.vpair.keys()):
            ecosts = []
            for c in range(self.NCFG(op_i)):
                for h in range(self.NCFG(op_j)):
                    ecosts.append(self.I[k][c, h] *
                                  self.PENALTY(op_i, c, op_j, h))
            ADD(self.cost[k] == cp.sum(ecosts))

        # 2> objective
        # start and end time of op i
        ADD(self.EO == self.SO + self.time)
        ADD(self.EE == self.SE + self.cost)
        for k, (op_i, op_j) in enumerate(self.vpair.keys()):
            ADD(self.SE[k] >= self.EO[op_i])
            ADD(self.SO[op_j] >= self.EE[k])
        # start and end time of device j
        for i in range(n):
            for j in range(N):
                ADD(self.SD[j] <= self.SO[i] + INF * (1 - self.M[i, j]))
                ADD(self.ED[j] >= self.EO[i] - INF * (1 - self.M[i, j]))
        # start and end time of stage s
        for s in range(S):
            for j in range(N):
                ADD(self.SS[s] <= self.SD[j] + INF * (1 - self.W[j, s]))
                ADD(self.ES[s] >= self.ED[j] - INF * (1 - self.W[j, s]))
        # max stage exe time
        for s in range(S):
            ADD(self.O >= self.ES[s] - self.SS[s])

        # 4> resource constraint
        # stage cannot be overlapped
        for k, (si, sj) in enumerate(spair):
            ADD(self.SS[si] >= self.ES[sj] - INF * self.U[k])
            ADD(self.SS[sj] >= self.ES[si] - INF * (1 - self.U[k]))

        print('Total {} constraints.'.format(ncstrs))
        return constrs

    def NCFG(self, op_id):
        return self.cfg_oracle.ncfg(op_id)

    def DEG(self, op_id, c):
        return self.cfg_oracle.deg(op_id, c)

    def COST(self, op_id, c=None):
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
                exe_time = INF / 10
        return exe_time

    def PENALTY(self, op_i, c, op_j, h):
        iparts = self.cfg_oracle.config(op_i, c)
        jparts = self.cfg_oracle.config(op_j, h)
        out_id, in_id = self.vpair[(op_i, op_j)]
        dshape = self.graph.ops[op_i].outs[out_id]
        di_parts, dj_parts = [], []
        for idx in self.graph.ops[op_i].iter_space.out_iters[out_id]:
            di_parts.append(iparts[idx])
        for idx in self.graph.ops[op_j].iter_space.in_iters[in_id]:
            dj_parts.append(jparts[idx])
        if di_parts == dj_parts:
            return 0
        acc = 1
        for i, (di, dj) in enumerate(zip(di_parts, dj_parts)):
            acc = acc * dshape[i] * max(di, dj) // min(di, dj)
        cost = acc * 4 / 1e6 / 50 + 0.005
        return cost

    def optimize(self, verbose=True):
        self.problem.solve(verbose=verbose, solver=cp.CPLEX, cplex_filename='log/model.lp')
