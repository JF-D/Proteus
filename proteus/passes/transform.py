from functools import lru_cache
import time
import queue
import numpy as np
from collections import OrderedDict, defaultdict
from proteus import TaskType, OpType, MapType, size_of_datatype, enum_to_str
from proteus.ir.tensor import Gradient, Input, Output, Parameter, Buffer
from proteus.simulator.cost_model import OpCostModel
from proteus.type import DevType
from proteus.utils import get_dst_part_groups, get_strides

kDefaultFirstBucketBytes = 1  # 1MB


class Task:
    id = 0
    __refs = {}

    def __init__(self, type, name):
        super().__init__()
        self.id = Task.id
        Task.id += 1
        self.type = type
        self.name = name

        self.next_tasks = []
        self.prev_tasks = []
        self.callbacks = []
        self.release_hooks = defaultdict(list)
        self.state = 0

        self.n_states = 1
        self._ndeps = defaultdict(lambda: len(self.prev_tasks))
        self.ready = defaultdict(lambda: 0)
        self._start = defaultdict(lambda: 0)
        self._end = defaultdict(lambda: 0)

        self.force_once = False  # for gradient comm in multi batch training

        self.recompute_task = None  # mainly for CompTask and MemTask
        self.attached_recompute_task = set()  # mainly comm task inserted
        self.is_recomputation = False
        Task.__refs[self.id] = self

        self.depth = -1

    @property
    def ndeps(self):
        return self._ndeps

    def reset(self):
        self.state = min(self.state + 1, self.n_states)
        self.done = False
        if self.type == TaskType.TASK_MEM and self.dev_id is None:
            self.dev_id = self.attach_dev

    def decrease_ndeps(self, state):
        self._ndeps[state] -= 1

    def dep_free(self, state=None):
        if self.force_once and self.state < self.n_states:
            return False
        if self.type == TaskType.TASK_MEM and self.persist:
            return True
        if state == None:
            return self.ndeps[self.state] == 0
        else:
            return self.ndeps[state] == 0

    def set_force_once(self):
        # only for leaf grad comm in multi macro batch
        self.force_once = True
        for tsk in self.next_tasks:
            tsk.force_once = True

    def set_n_max_state(self, n_macro_batch):
        self.n_states = n_macro_batch

    def add_next_task(self, task):
        self.next_tasks.append(task)
        task.prev_tasks.append(self)

    def remove_prev_task(self, task):
        self.prev_tasks.remove(task)

    def remove_next_task(self, task):
        self.next_tasks.remove(task)

    def add_callback(self, func):
        self.callbacks.append(func)

    @property
    def start(self):
        return self._start[self.state]

    @property
    def end(self):
        return self._end[self.state]

    def get_ready(self):
        return self.ready[self.state]

    def set_start_time(self, start_time):
        self._start[self.state] = start_time
        self._end[self.state] = start_time + self.cost
        self.done = True
        return self.end

    @staticmethod
    def get(index):
        return Task.__refs[index]

    @staticmethod
    def total_tasks():
        return len(Task.__refs)

    def __hash__(self) -> int:
        return self.id

    def __repr__(self):
        return 'f{}[{}]'.format(self.id, self.name)


def convert_dummy_device(dev):

    class Dev:

        def __init__(self, type):
            self.type = type

    if dev.startswith('gpu'):
        return Dev(DevType.GPU)
    elif dev.startswith('cpu'):
        return Dev(DevType.CPU)


class MemTask(Task):

    def __init__(self,
                 data,
                 index,
                 dev=None,
                 partial=False,
                 partial_group=None,
                 nelements=None,
                 permanent=False):
        if partial:
            name = '(p){}_{}/{}:{}'.format(data.name, data.id, index,
                                           dev if dev else -1)
        else:
            name = '{}_{}/{}:{}'.format(data.name, data.id, index,
                                        dev if dev else -1)
        super().__init__(TaskType.TASK_MEM, name)
        self.data = data
        self.index = index
        self.dev_id = dev
        self.persist = False

        self.cost = 0
        element_size = size_of_datatype(data.dtype) * (1 + data.buffer_account)
        if index < 0:
            self.memory = element_size * nelements
        else:
            self.memory = element_size * data.sub_tensors[index].nelements()
        self.memory = self.memory / 1024 / 1024
        if partial:
            self.memory = self.memory * len(partial_group)

        self.is_grad = isinstance(data, Gradient)
        self.is_leaf_grad = self.is_grad and data.is_leaf_grad
        self.permanent = permanent and not partial and (isinstance(
            data, (Input, Parameter, Buffer, Output)) or self.is_leaf_grad)
        self.first_round = True

        self.intermediate_data = None
        self.attach_dev = None

    @property
    def rank(self):
        if self.dev_id != -1:
            return int(self.dev_id.split(':')[1])
        return self.dev_id


class CompTask(Task):

    def __init__(self,
                 op,
                 index,
                 dev,
                 cost_type='profile',
                 reprofile=False,
                 profile_iters=10):
        name = '{}_{}/{}:{}'.format(enum_to_str(OpType, op.type), op.id, index,
                                    dev)
        super().__init__(TaskType.TASK_COMP, name)
        self.op = op
        self.index = index
        self.dev_id = dev
        self._rank = int(self.dev_id.split(':')[1])

        self.cost = self.measure_cost(dev, cost_type, reprofile, profile_iters)

    def measure_cost(self,
                     dev,
                     cost_type='profile',
                     reprofile=False,
                     profile_iters=10):
        op_ins, op_outs = self.op.ins, self.op.outs
        dev = convert_dummy_device(dev)

        self.op.ins = tuple([x.size for x in self.op.sub_ins[self.index]])
        self.op.outs = tuple([y.size for y in self.op.sub_outs[self.index]])
        cost = self.op.measure_cost(dev, cost_type, reprofile, profile_iters)
        self.op.ins = op_ins
        self.op.outs = op_outs
        return cost

    @property
    def rank(self):
        return self._rank


class CommTask(Task):

    def __init__(self,
                 volume,
                 src,
                 dst,
                 collective=None,
                 is_grad_comm=False,
                 comm_type='p2p',
                 group=None):
        if collective is not None:
            name = '{}_{}'.format(collective, Task.id)
        elif comm_type == 'cpu2gpu':
            name = 'H2D_{}'.format(Task.id)
        elif comm_type == 'gpu2cpu':
            name = 'D2H_{}'.format(Task.id)
        else:
            name = 'xfer_{}/{}_{}'.format(Task.id, src, dst)
        super().__init__(TaskType.TASK_COMM, name)
        self.volume = volume
        self.src = src if collective is None else None
        self.dst = dst if collective is None else None
        self.is_grad_comm = is_grad_comm
        self.is_cross_stage = False

        self.collective = collective is not None
        self.group = [src, dst] if group is None else group
        self.comm_type = comm_type
        self.overlap_factor = 0

    def set_cost(self, machine_model, type_intra, type_inter, cross_node):
        self.bw = machine_model[0]
        self.lat = machine_model[1]
        self.cost = (self.lat + self.volume * 1e3 / self.bw) * 1e-3
        if self.comm_type == 'all_to_all':
            self.cost = self.cost * (len(self.group) - 1)
        self.type_intra = type_intra
        self.type_inter = type_inter
        self.cross_node = cross_node
        self.share_factor = 1

    def set_share_bw_cost(self, share_bw):
        self.cost = (self.lat + self.volume * 1e3 / share_bw) * 1e-3
        self.cost = self.cost * (1 + self.overlap_factor)
        if self.comm_type == 'all_to_all':
            self.cost = self.cost * (len(self.group) - 1)

    @staticmethod
    def make_task_like(task, volume=None):
        volume = task.volume if volume is None else volume
        collective = task.name.split('_')[1] if task.collective else None
        new_task = CommTask(volume,
                            task.src,
                            task.dst,
                            collective=collective,
                            is_grad_comm=task.is_grad_comm,
                            comm_type=task.comm_type,
                            group=task.group)
        return new_task


class TaskGraph:
    id = 0

    def __init__(self,
                 is_first_stage=False,
                 cost_type='roofline',
                 reprofile=False,
                 profile_iters=10,
                 forward=True,
                 optimizer=False):
        self.is_first_stage = is_first_stage
        self.cost_type = cost_type
        self.reprofile = reprofile
        self.profile_iters = profile_iters
        self.forward = forward
        self.is_optimizer = optimizer

        self.id = TaskGraph.id
        TaskGraph.id += 1

        self.ops = OrderedDict()
        self.datas = {}
        self.ins = []
        self.outs = []

        self.task_ids = set()
        self.dev_ids = set()

        self.next = []
        self.ongoing_control = []
        self.callbacks = []

        self.n_macro_batch = 1
        self.interleave_freq = 1
        self.max_ongoing_macro_batch = 1
        self.cur_macro_batch = 0
        self.last_ongoing_batch = 0
        self.max_available_batch = self.n_macro_batch if self.is_first_stage else 0
        self._dep_free = self.is_first_stage

    def set_schedule(self, vnode):
        self.n_macro_batch = vnode.pconfig['schedule']['n_macro_batch']
        self.interleave_freq = vnode.pconfig['schedule']['interleave_freq']
        self.max_ongoing_macro_batch = vnode.pconfig['schedule'][
            'max_ongoing_macro_batch']
        self.max_available_batch = self.n_macro_batch if self.is_first_stage else 0

    def add_next(self, ngraph):
        for data in ngraph.ins:
            assert len(data.producer) == 0 or data in self.outs
        self.next.append(ngraph)

    def set_forward(self, ngraph):
        self.ongoing_control.append(ngraph)

    def set_stage_id(self, stage_id):
        self.id = stage_id

    def macro_batch_done_debug(self):
        while len(self.batch_data_tasks) > 0:
            tsk = self.batch_data_tasks.pop()
            if (not tsk.done and tsk.id in self.task_ids) or (
                    tsk.id not in self.task_ids and not tsk.dep_free()):
                print('>>> batch 0: ', tsk, tsk.ndeps, tsk.prev_tasks)
                for ptask in tsk.prev_tasks:
                    print(' ' * 4, ptask, ptask.done, ptask.ndeps)
                self.batch_data_tasks.add(tsk)
                return False

        if len(self.batch_data_tasks) == 0:
            while len(self.batch_tasks) > 0:
                tsk = self.batch_tasks.pop()
                if not tsk.done:
                    self.batch_tasks.add(tsk)
                    print('>>> batch 1: ', tsk, tsk.is_recomputation)
                    return False
        return True

    def macro_batch_done(self):
        while len(self.batch_data_tasks) > 0:
            tsk = self.batch_data_tasks.pop()
            if (not tsk.done and tsk.id in self.task_ids) or (
                    tsk.id not in self.task_ids and not tsk.dep_free(self.cur_macro_batch + 1)):
                self.batch_data_tasks.add(tsk)
                return False

        if len(self.batch_data_tasks) == 0:
            while len(self.batch_tasks) > 0:
                tsk = self.batch_tasks.pop()
                if not tsk.done:
                    self.batch_tasks.add(tsk)
                    return False
        return True

    def done(self):
        return self.cur_macro_batch >= self.n_macro_batch

    def grad_not_done_only(self):
        while len(self.not_grad_task_ids) > 0:
            tid = self.not_grad_task_ids.pop()
            tsk = Task.get(tid)
            if not tsk.done:
                self.not_grad_task_ids.add(tid)
                return False
        return True

    def dep_free(self):
        if self.done():
            return False
        if self.cur_macro_batch >= self.max_available_batch:
            return False
        if self.cur_macro_batch - self.last_ongoing_batch < self.max_ongoing_macro_batch:
            return True
        return False

    def reset(self, state=None):
        for tid in self.task_ids:
            Task.get(tid).reset()
            if state is not None:
                Task.get(tid).state = state

        self.batch_data_tasks = set()
        # for data in self.outs:
        #     for _, sub_tasks in data.sub_tasks.items():
        #         for tsk in sub_tasks:
        #             if tsk.force_once and self.cur_macro_batch < self.n_macro_batch - 1:
        #                 continue
        #             self.batch_data_tasks.add(tsk)

        self.batch_tasks = set()
        if len(self.batch_data_tasks) == 0:
            for tsk_id in self.task_ids:
                task = Task.get(tsk_id)
                if task.force_once and self.cur_macro_batch < self.n_macro_batch - 1:
                    continue
                self.batch_tasks.add(task)

        self.not_grad_task_ids = set()
        for tid in self.task_ids:
            tsk = Task.get(tid)
            if tsk.type == TaskType.TASK_MEM and tsk.is_leaf_grad:
                continue
            elif tsk.type == TaskType.TASK_COMM and tsk.is_grad_comm:
                continue
            self.not_grad_task_ids.add(tid)

    def execute(self):

        def callback():
            for tgraph in self.next:
                tgraph.max_available_batch += 1
            for tgraph in self.ongoing_control:
                tgraph.last_ongoing_batch += 1
            if not self.forward:
                self.last_ongoing_batch += 1

        self.cur_macro_batch += 1
        self.callbacks.append(callback)
        # reset statistics
        if not self.done():
            self.reset()

    def create_tasks(self, graph, stree):
        for op_id, op in self.ops.items():
            assert getattr(op, 'sub_tasks', None) is None
            config = graph.op_config[op_id]
            op.partition(config)

            recomputation = False
            if self.forward:
                for i, data in enumerate(op.write):
                    datacfg = graph.data_config[data.id]
                    if datacfg.is_recompute():
                        recomputation = True
                        break
            op.recomputation = recomputation

            op.sub_tasks = {}
            for i in range(config.deg()):
                mapping = config.mapping[i]
                tasks = []
                for dev_id in mapping[1:]:
                    task = CompTask(op, i, stree.dev_topo.dev(dev_id),
                                    self.cost_type, self.reprofile,
                                    self.profile_iters)
                    tasks.append(task)
                    self.task_ids.add(task.id)
                    if recomputation:
                        rc_tsk = CompTask(op, i, stree.dev_topo.dev(dev_id),
                                          self.cost_type, False)
                        rc_tsk.is_recomputation = True
                        task.recompute_task = rc_tsk
                op.sub_tasks[i] = tasks

        for data_id, data in self.datas.items():
            assert getattr(data, 'sub_tasks', None) is None
            config = graph.data_config[data_id]
            data.partition(config)
            data.sub_tasks = {}
            for i in range(config.deg()):
                mapping = config.mapping[i]
                tasks = []
                for dev_id in mapping[1:]:
                    task = MemTask(data, i, stree.dev_topo.dev(dev_id), permanent=True)
                    tasks.append(task)
                    task.persist = data._is_parameter
                    self.task_ids.add(task.id)
                    if config.is_recompute():
                        rc_task = MemTask(data, i, stree.dev_topo.dev(dev_id))
                        rc_task.is_recomputation = True
                        task.recompute_task = rc_task

                data.sub_tasks[i] = tasks

    def all_reduce_collective_comm(self, op, opcfg, wid, data, datacfg,
                                   dev_topo, recompute=False):
        data_parts, data_map, data_mesh = datacfg.get_config()
        op_parts, op_map, op_mesh = op.infer_output_config(wid, opcfg)
        need_reduce = (np.prod(op_parts) != opcfg.deg())
        if data_parts == op_parts and data_map == op_map:
            if need_reduce:
                is_grad_comm = isinstance(data, Gradient) and data.is_leaf_grad

                for rep in range(opcfg.replicate_degree):
                    shard_devs = [
                        rep_map[rep + 1] for rep_map in opcfg.mapping
                    ]
                    for i in range(len(op_map)):
                        assert op_map[i][0] == MapType.REPLICATE
                        ptasks, group = [], []
                        for j, dev_id in enumerate(op_map[i][1:]):
                            if dev_id not in shard_devs:
                                continue
                            group.append(dev_id)
                            # partial gradient memory task
                            ptask = MemTask(data,
                                            i,
                                            dev_topo.dev(dev_id),
                                            partial=True,
                                            partial_group=[i])
                            idx = shard_devs.index(dev_id)
                            if recompute:
                                op.sub_tasks[idx][rep].recompute_task.add_next_task(ptask)
                            else:
                                op.sub_tasks[idx][rep].add_next_task(ptask)
                                self.task_ids.add(ptask.id)
                            ptasks.append((j, ptask))

                        all_reduce_task = CommTask(
                            ptask.memory * (1.024**2),
                            None,
                            None,
                            collective='{}-{}_allred'.format(data, i),
                            is_grad_comm=is_grad_comm,
                            comm_type='all_reduce',
                            group=group)
                        if not recompute:
                            self.task_ids.add(all_reduce_task.id)
                        for (idx, ptask) in ptasks:
                            ptask.add_next_task(all_reduce_task)
                            dst_task = data.sub_tasks[i][idx]
                            if recompute:
                                all_reduce_task.add_next_task(dst_task.recompute_task)
                                dst_task.recompute_task.attached_recompute_task.update([
                                    ptask.id, all_reduce_task.id
                                ])
                            else:
                                all_reduce_task.add_next_task(dst_task)
                return True
            else:
                for idx in range(opcfg.deg()):
                    op_sts = op.sub_tasks[idx]
                    data_sts = data.sub_tasks[idx]
                    for src_tsk, dst_tsk in zip(op_sts, data_sts):
                        if recompute:
                            src_tsk.recompute_task.add_next_task(dst_tsk.recompute_task)
                        else:
                            src_tsk.add_next_task(dst_tsk)
                return True
        return False

    def reduce_scatter_collective_comm(self, op, opcfg, wid, data, datacfg,
                                       dev_topo, recompute=False):
        data_parts, data_map, data_mesh = datacfg.get_config()
        op_parts, op_map, op_mesh = op.infer_output_config(wid, opcfg)
        need_reduce = (np.prod(op_parts) != opcfg.deg())

        if not need_reduce:
            return False
        group = get_dst_part_groups(data_parts, datacfg.strides, op_parts,
                                    get_strides(op_parts))
        if group is None:
            return False
        for i, gps in group.items():
            if op_map[i][0] != MapType.REPLICATE:
                return False
            src_devs = list(op_map[i][1:])
            dst_devs = []

            map_len = len(data_map[gps[0]])
            for j in gps:
                if len(data_map[j]) != map_len:
                    return False
                dst_devs.extend(data_map[j][1:])
            if sorted(dst_devs) != sorted(src_devs):
                return False
            if map_len > 2:
                # output is replicated
                for j in gps:
                    if data_map[j] not in opcfg.mapping:
                        return False

        opcfg_map = list(opcfg.mesh.reshape(-1))
        is_grad_comm = isinstance(data, Gradient) and data.is_leaf_grad
        replicate_deg = opcfg.mesh.shape[-1]
        for i, gps in group.items():
            for rep in range(len(data_map[gps[0]]) - 1):
                dev_group = []
                ptasks = []
                for dst_id in gps:
                    dev_id = data_map[dst_id][rep + 1]
                    dev_group.append(dev_id)

                    ptask = MemTask(data,
                                    dst_id,
                                    dev_topo.dev(dev_id),
                                    partial=True,
                                    partial_group=gps)

                    map_index = opcfg_map.index(dev_id)
                    deg_id = map_index // replicate_deg
                    rep_id = map_index % replicate_deg
                    assert opcfg.mapping[deg_id][rep_id + 1] == dev_id
                    src_task = op.sub_tasks[deg_id][rep_id]
                    if recompute:
                        src_task.recompute_task.add_next_task(ptask)
                    else:
                        src_task.add_next_task(ptask)
                        self.task_ids.add(ptask.id)
                    ptasks.append(ptask)

                red_sca_task = CommTask(ptask.memory * (1.024**2),
                                        None,
                                        None,
                                        collective='{}-{}_redsca'.format(data, i),
                                        is_grad_comm=is_grad_comm,
                                        comm_type='reduce_scatter',
                                        group=dev_group)
                if not recompute:
                    self.task_ids.add(red_sca_task.id)
                for k, ptask in enumerate(ptasks):
                    ptask.add_next_task(red_sca_task)
                    dst_task = data.sub_tasks[gps[k]][rep]
                    if recompute:
                        red_sca_task.add_next_task(dst_task.recompute_task)
                        dst_task.recompute_task.attached_recompute_task.update([
                            ptask.id, red_sca_task.id
                        ])
                    else:
                        red_sca_task.add_next_task(dst_task)
        return True

    def all_gather_write_collective_comm(self, op, opcfg, wid, data, datacfg,
                                         dev_topo, recompute=False):
        data_parts, data_map, data_mesh = datacfg.get_config()
        op_parts, op_map, op_mesh = op.infer_output_config(wid, opcfg)
        if data_parts != op_parts or data_map != op_map:
            group = get_dst_part_groups(op_parts, get_strides(op_parts),
                                        data_parts, datacfg.strides)
            if group is None:
                return False
            for i, gps in group.items():
                if data_map[i][0] != MapType.REPLICATE:
                    return False
                dst_devs = list(data_map[i][1:])
                src_devs = []
                map_len = len(op_map[gps[0]])
                for j in gps:
                    if len(op_map[j]) != map_len:
                        return False
                    src_devs.extend(op_map[j][1:])
                if sorted(dst_devs) != sorted(src_devs):
                    return False

            is_grad_comm = isinstance(data, Gradient) and data.is_leaf_grad

            for i, gps in group.items():
                volume = data.sub_tasks[i][0].memory * (1.024**2)
                for rep in range(len(op_map[gps[0]]) - 1):
                    dev_group = []
                    for j in gps:
                        dev_group.append(op_map[j][rep + 1])
                    all_gather_task = CommTask(volume,
                                               None,
                                               None,
                                               collective='{}-{}_allgat'.format(
                                                   data, i),
                                               is_grad_comm=is_grad_comm,
                                               comm_type='all_gather',
                                               group=dev_group)
                    if not recompute:
                        self.task_ids.add(all_gather_task.id)
                    for src_id in gps:
                        src_task = op.sub_tasks[src_id][rep]
                        # intermediata data
                        inter_mem = MemTask(data, i, dev=src_task.dev_id)
                        if recompute:
                            src_task.recompute_task.add_next_task(inter_mem)
                        else:
                            self.task_ids.add(inter_mem.id)
                            src_task.add_next_task(inter_mem)
                        inter_mem.add_next_task(all_gather_task)

                        dst_dev = op_map[src_id][rep + 1]
                        k = data_map[i].index(dst_dev) - 1
                        dst_task = data.sub_tasks[i][k]
                        if recompute:
                            all_gather_task.add_next_task(dst_task.recompute_task)
                            dst_task.recompute_task.attached_recompute_task.update([
                                inter_mem.id, all_gather_task.id
                            ])
                        else:
                            all_gather_task.add_next_task(dst_task)
            return True
        return False

    def all_gather_collective_comm(self, op, opcfg, rid, data, datacfg,
                                   dev_topo, recompute=False):
        data_parts, data_map, data_mesh = datacfg.get_config()
        op_parts, op_map, op_mesh = op.infer_input_config(rid, opcfg)
        if data_parts != op_parts or data_map != op_map:
            group = get_dst_part_groups(data_parts, datacfg.strides, op_parts,
                                        get_strides(op_parts))
            if group is None:
                return False
            for i, gps in group.items():
                if op_map[i][0] != MapType.REPLICATE:
                    return False
                dst_devs = list(op_map[i][1:])
                src_devs = []
                map_len = len(data_map[gps[0]])
                for j in gps:
                    if len(data_map[j]) != map_len:
                        return False
                    src_devs.extend(data_map[j][1:])
                if sorted(dst_devs) != sorted(src_devs):
                    return False

            is_grad_comm = isinstance(data, Gradient) and data.is_leaf_grad
            can_be_prefetch = self.can_be_prefetched(op, rid, from_dev='gpu')

            recompute_bw = datacfg.is_recompute() and not self.forward and not self.is_optimizer

            opcfg_map = list(opcfg.mesh.reshape(-1))
            replicate_deg = opcfg.mesh.shape[-1]
            for i, gps in group.items():
                volume = data.sub_tasks[gps[0]][0].memory * (1.024**
                                                             2) * len(gps)
                for rep in range(len(data_map[gps[0]]) - 1):
                    dev_group = []
                    for j in gps:
                        dev_group.append(data_map[j][rep + 1])
                    all_gather_task = CommTask(volume,
                                               None,
                                               None,
                                               collective='{}-{}_allgat'.format(
                                                   data, i),
                                               is_grad_comm=is_grad_comm,
                                               comm_type='all_gather',
                                               group=dev_group)
                    if not recompute:
                        self.task_ids.add(all_gather_task.id)
                    for src_id in gps:
                        src_task = data.sub_tasks[src_id][rep]
                        if recompute_bw or (recompute and src_task.recompute_task is not None):
                            src_task.recompute_task.add_next_task(all_gather_task)
                        else:
                            src_task.add_next_task(all_gather_task)

                        dst_dev = data_map[src_id][rep + 1]
                        map_index = opcfg_map.index(dst_dev)
                        deg_id = map_index // replicate_deg
                        rep_id = map_index % replicate_deg
                        assert opcfg.mapping[deg_id][rep_id + 1] == dst_dev
                        dst_task = op.sub_tasks[deg_id][rep_id]

                        # intermedaita data
                        inter_mem = MemTask(data, src_id, dst_task.dev_id)
                        if not recompute:
                            self.task_ids.add(inter_mem.id)

                        all_gather_task.add_next_task(inter_mem)
                        if recompute:
                            inter_mem.add_next_task(dst_task.recompute_task)
                        else:
                            inter_mem.add_next_task(dst_task)

                    if can_be_prefetch:
                        op.add_prefetch_task(all_gather_task,
                                             rid,
                                             all_gather_task.group,
                                             from_dev='gpu')
            return True
        return False

    def scatter_comm(self, op, opcfg, wid, data, datacfg, dev_topo):
        data_parts, data_map, data_mesh = datacfg.get_config()
        op_parts, op_map, op_mesh = op.infer_output_config(wid, opcfg)
        # temporary for DLRM
        if opcfg.deg() == 1 and len(op_map[0]) == 2 and datacfg.deg() > 1:
            # inter_mem = MemTask(data, 0, dst_task.dev_id)
            inter_mem = MemTask(data, -1, dev_topo.dev(op_map[0][1]),
                                nelements=op.sub_outs[0][wid].nelements())
            self.task_ids.add(inter_mem.id)

            volume = data.sub_tasks[0][0].memory * (1.024**2)
            is_grad_comm = isinstance(data, Gradient) and data.is_leaf_grad
            dev_group = [dm[1] for dm in data_map]
            src = dev_topo.dev(op_map[0][1])
            scatter_task = CommTask(volume,
                                    None,
                                    None,
                                    collective='{}-{}_scatter'.format(
                                        data, src),
                                    is_grad_comm=is_grad_comm,
                                    comm_type='scatter',
                                    group=dev_group)
            self.task_ids.add(scatter_task.id)

            op.sub_tasks[0][0].add_next_task(inter_mem)
            inter_mem.add_next_task(scatter_task)
            for dst_id, dst_tsks in data.sub_tasks.items():
                scatter_task.add_next_task(dst_tsks[0])
            return True
        return False

    def gather_comm(self, op, opcfg, wid, data, datacfg, dev_topo):
        data_parts, data_map, data_mesh = datacfg.get_config()
        op_parts, op_map, op_mesh = op.infer_output_config(wid, opcfg)
        # temporary for DLRM
        if opcfg.deg() > 1 and len(data_map[0]) == 2 and datacfg.deg() == 1:
            volume = data.sub_tasks[0][0].memory * (1.024**2) / opcfg.deg()
            is_grad_comm = isinstance(data, Gradient) and data.is_leaf_grad
            dev_group = [dm[1] for dm in op_map]
            dst = dev_topo.dev(data_map[0][1])
            gather_task = CommTask(volume,
                                   None,
                                   None,
                                   collective='{}-{}_gather'.format(
                                       data, dst),
                                   is_grad_comm=is_grad_comm,
                                   comm_type='gather',
                                   group=dev_group)
            self.task_ids.add(gather_task.id)

            for i in range(opcfg.deg()):
                inter_mem = MemTask(data, -1, dev=dev_topo.dev(op_map[i][1]),
                                    nelements=op.sub_outs[i][wid].nelements())
                self.task_ids.add(inter_mem.id)
                op.sub_tasks[i][0].add_next_task(inter_mem)
                inter_mem.add_next_task(gather_task)
            gather_task.add_next_task(data.sub_tasks[0][0])
            return True
        return False


    def add_recompute_dep(self, task):
        self.task_ids.add(task.id)
        q = queue.Queue()
        for tsk in task.prev_tasks:
            if tsk.is_recomputation:
                q.put(tsk)
        while not q.empty():
            tsk = q.get()
            self.task_ids.add(tsk.id)
            for ptsk in tsk.prev_tasks:
                if ptsk.is_recomputation:
                    q.put(ptsk)

    def check_inter_cpu_gpu(self, src_dev_type, dst_dev_type):
        if src_dev_type is None or dst_dev_type is None:
            return None
        if src_dev_type == dst_dev_type:
            return None
        return f'{src_dev_type}2{dst_dev_type}'

    def can_be_prefetched(self, op, rid, from_dev='gpu'):
        # prefetch_depth = 1, can be prefetch from gpu
        # prefetch_depth = 2, can be prefetch from cpu
        prefetch_depth = 1 if from_dev == 'gpu' else 2

        producer_depth = [-1]
        for (prev_op, _) in op.read[rid].producer:
            producer_depth.append(prev_op.depth)
        data_depth = 1 + max(producer_depth)
        if data_depth == 0 or data_depth + prefetch_depth < op.depth:
            return True
        return False

    def cpu_to_gpu(self, data, datacfg, dev_topo, op, rid):
        can_be_prefetch = self.can_be_prefetched(op, rid, from_dev='cpu')
        fake_mapping, fake_tasks = [], {}
        for i in range(datacfg.deg()):
            mapping = datacfg.mapping[i]
            tasks = []
            fake_map = [mapping[0]]
            if mapping[0] == MapType.REPLICATE:
                for dev_id in mapping[1:]:
                    dev_id = 'gpu:' + dev_id.split(':')[1]
                    fake_map.append(dev_id)
                    task = MemTask(data, i, dev_topo.dev(dev_id))
                    tasks.append(task)
                    self.task_ids.add(task.id)
            elif mapping[0] == MapType.SHARD:
                dev_id = 'gpu:' + mapping[1].split(':')[1]
                fake_map.append(dev_id)
                task = MemTask(data, i, dev_topo.dev(dev_id))
                tasks.append(task)
                self.task_ids.add(task.id)
            else:
                assert False
            fake_mapping.append(fake_map)
            fake_tasks[i] = tasks

            # add comm dep
            for k, (src_tsk,
                    dst_tsk) in enumerate(zip(data.sub_tasks[i],
                                              fake_tasks[i])):
                volume = src_tsk.memory * (1.024**2)
                comm = CommTask(volume,
                                mapping[k + 1],
                                fake_map[k + 1],
                                comm_type='cpu2gpu')
                src_tsk.add_next_task(comm)
                comm.add_next_task(dst_tsk)
                self.task_ids.add(comm.id)

                if can_be_prefetch:
                    op.add_prefetch_task(comm,
                                         rid, [fake_map[k + 1]],
                                         from_dev='cpu')
        fake_mesh = []
        mesh_shape = datacfg.mesh.shape
        for dev in datacfg.mesh.reshape(-1):
            assert dev.startswith('cpu:')
            fake_mesh.append('gpu:' + dev[4:])
        fake_mesh = np.array(fake_mesh).reshape(mesh_shape)
        return fake_tasks, fake_mapping, fake_mesh

    def gpu_to_cpu(self, data, datacfg, dev_topo):
        fake_mapping, fake_tasks = [], {}
        for i in range(datacfg.deg()):
            mapping = datacfg.mapping[i]
            tasks = []
            fake_map = [mapping[0]]
            if mapping[0] == MapType.REPLICATE:
                for dev_id in mapping[1:]:
                    dev_id = 'gpu:' + dev_id.split(':')[1]
                    fake_map.append(dev_id)
                    task = MemTask(data, i, dev_topo.dev(dev_id))
                    tasks.append(task)
                    self.task_ids.add(task.id)
            elif mapping[0] == MapType.SHARD:
                dev_id = 'gpu:' + mapping[1].split(':')[1]
                fake_map.append(dev_id)
                task = MemTask(data, i, dev_topo.dev(dev_id))
                tasks.append(task)
                self.task_ids.add(task.id)
            else:
                assert False
            fake_mapping.append(fake_map)
            fake_tasks[i] = tasks

            # add comm dep
            for k, (dst_tsk,
                    src_tsk) in enumerate(zip(data.sub_tasks[i],
                                              fake_tasks[i])):
                volume = src_tsk.memory * (1.024**2)
                comm = CommTask(volume,
                                mapping[k + 1],
                                fake_map[k + 1],
                                comm_type='gpu2cpu')
                src_tsk.add_next_task(comm)
                comm.add_next_task(dst_tsk)
                self.task_ids.add(comm.id)

        fake_mesh = []
        mesh_shape = datacfg.mesh.shape
        for dev in datacfg.mesh.reshape(-1):
            assert dev.startswith('cpu:')
            fake_mesh.append('gpu:' + dev[4:])
        fake_mesh = np.array(fake_mesh).reshape(mesh_shape)
        return fake_tasks, fake_mapping, fake_mesh

    def read_share_gradient(self, data, datacfg, graph):
        # shared data optimization --- begin
        # data_parts, data_map, data_mesh = datacfg.get_config()
        # for i, dst_tasks in data.sub_tasks.items():
        #     all_reduce_task = CommTask(dst_tasks[0].memory * (1.024**2),
        #                                None,
        #                                None,
        #                                collective='{}-{}_allred'.format(
        #                                    data, i),
        #                                is_grad_comm=True,
        #                                comm_type='all_reduce',
        #                                group=data_map[i][1:])
        #     self.task_ids.add(all_reduce_task.id)
        #     for share_grad in data.share_params:
        #         for src_tsk in share_grad.sub_tasks[i]:
        #             src_tsk.add_next_task(all_reduce_task)
        #     for dst_tsk in dst_tasks:
        #         all_reduce_task.add_next_task(dst_tsk)
        # shared data optimization --- end
        share_data_cfgs = []
        for share_grad in data.share_params:
            share_data_cfg = graph.data_config[share_grad.id]
            share_data_cfgs.append(share_data_cfg)
        for cfg in share_data_cfgs:
            assert tuple(cfg.parts) == (share_data_cfgs[0].parts)
            assert cfg.mesh.shape == share_data_cfgs[0].mesh.shape
        for idx in range(len(data.share_params[0].sub_tasks)):
            dst_tasks = data.sub_tasks[idx]
            dst_map = datacfg.mapping[idx][1:]
            for rep_id in range(len(data.share_params[0].sub_tasks[idx])):
                src_tasks, group = [], []
                for k, share_grad in enumerate(data.share_params):
                    src_tasks.append(share_grad.sub_tasks[idx][rep_id])
                    group.append(share_data_cfgs[k].mapping[idx][rep_id + 1])

                all_reduce_task = CommTask(src_tasks[0].memory * (1.024**2),
                                           None,
                                           None,
                                           collective='{}-{}_allred'.format(
                                               data, idx),
                                           is_grad_comm=True,
                                           comm_type='all_reduce',
                                           group=group)
                self.task_ids.add(all_reduce_task.id)
                for src_tsk, dev_id in zip(src_tasks, group):
                    src_tsk.add_next_task(all_reduce_task)
                    dst_tsk = dst_tasks[dst_map.index(dev_id)]
                    all_reduce_task.add_next_task(dst_tsk)

    def read_recompute_data(self, data):
        if data.sub_tasks[0][0].recompute_task is None:
            return
        if data.sub_tasks[0][0].recompute_task in self.task_ids:
            return
        for _, sub_tasks in data.sub_tasks.items():
            for tsk in sub_tasks:
                self.task_ids.add(tsk.recompute_task.id)
                for tid in tsk.recompute_task.attached_recompute_task:
                    self.task_ids.add(tid)
        for op, _ in data.producer:
            self.read_recompute_op(op)

    def read_recompute_op(self, op):
        if not op.recomputation:
            return
        if op.sub_tasks[0][0].id in self.task_ids:
            return
        for _, sub_tasks in op.sub_tasks.items():
            for tsk in sub_tasks:
                self.task_ids.add(tsk.recompute_task.id)
                for tid in tsk.recompute_task.attached_recompute_task:
                    self.task_ids.add(tid)
        for data in op.read:
            self.read_recompute_data(data)

    def op_read_data(self, op, rid, data, opcfg, datacfg, stree, graph, recompute=False):
        if self.is_optimizer and data.is_shared and isinstance(data, Gradient):
            data_parts, data_map, data_mesh = datacfg.get_config()
            op_parts, op_map, op_mesh = op.infer_input_config(rid, opcfg)
            assert op_parts == data_parts
            assert op_map == data_map
            self.read_share_gradient(data, datacfg, graph)

        recompute_bw = datacfg.is_recompute() and not self.forward and not self.is_optimizer
        if recompute_bw:
            self.read_recompute_data(data)
        if stree.collective_comm and self.all_gather_collective_comm(
                op, opcfg, rid, data, datacfg, stree.dev_topo, recompute=recompute):
            return
        for dst_id in range(opcfg.deg()):
            dst_interval = op.sub_ins[dst_id][rid]
            for src_id in range(datacfg.deg()):
                src_interval = data.sub_tensors[src_id]
                area = dst_interval.intersection(src_interval)
                if area <= 0:
                    continue

                rep_equal = len(op.sub_tasks[dst_id]) == len(data.sub_tasks[src_id])
                for i, dst_task in enumerate(op.sub_tasks[dst_id]):
                    if rep_equal:
                        src_task = data.sub_tasks[src_id][i]
                    else:
                        bw = -999
                        for src_task_c in data.sub_tasks[src_id]:
                            if src_task_c.intermediate_data is not None and op.depth == data.producer[
                                    0][0].depth + 1:
                                src_tc = src_task_c.intermediate_data
                            else:
                                src_tc = src_task_c

                            if stree.dev_topo.bw(src_tc.dev_id,
                                                 dst_task.dev_id) > bw:
                                bw = stree.dev_topo.bw(src_tc.dev_id,
                                                       dst_task.dev_id)
                                src_task = src_tc
                    if src_task.dev_id == dst_task.dev_id:
                        if recompute_bw:
                            src_task.recompute_task.add_next_task(dst_task)
                        elif recompute:
                            if src_task.recompute_task is None:
                                src_task.add_next_task(dst_task.recompute_task)
                            else:
                                src_task.recompute_task.add_next_task(dst_task.recompute_task)
                        else:
                            src_task.add_next_task(dst_task)
                        continue

                    volume = area * size_of_datatype(data.dtype) / 1e6
                    task = CommTask(volume,
                                    stree.dev_topo.dev(src_task.dev_id),
                                    stree.dev_topo.dev(dst_task.dev_id))
                    # make intemediate data
                    inter_mem = MemTask(data, src_id,
                                        stree.dev_topo.dev(src_task.dev_id))
                    if recompute:
                        if src_task.recompute_task is not None:
                            src_task.recompute_task.attached_recompute_task.update([
                                task.id, inter_mem.id
                            ])
                        else:
                            dst_task.recompute_task.attached_recompute_task.update([
                                task.id, inter_mem.id
                            ])
                    else:
                        self.task_ids.add(task.id)
                        self.task_ids.add(inter_mem.id)

                    if recompute_bw:
                        src_task.recompute_task.add_next_task(task)
                    elif recompute:
                        if src_task.recompute_task is None:
                            src_task.add_next_task(task)
                        else:
                            src_task.recompute_task.add_next_task(task)
                    else:
                        src_task.add_next_task(task)
                    task.add_next_task(inter_mem)
                    if recompute:
                        inter_mem.add_next_task(dst_task.recompute_task)
                    else:
                        inter_mem.add_next_task(dst_task)

    def op_write_data(self, op, wid, data, opcfg, datacfg, stree, recompute=False):
        # shared data optimization --- begin
        # write shared data
        # if data.is_share_data:
        #     op_parts, op_map, op_mesh = op.infer_output_config(wid, opcfg)
        #     data_parts, data_map, data_mesh = datacfg.get_config()
        #     assert op_parts == data_parts
        #     assert op_map == data_map
        #     assert opcfg.is_shard
        #     for dst_id in data.sub_tasks:
        #         for dst_tsk in data.sub_tasks[dst_id]:
        #             idx = opcfg.mapping.index((MapType.SHARD, dst_tsk.dev_id))
        #             op.sub_tasks[idx][0].add_next_task(dst_tsk)
        #     return
        # shared data optimization --- end
        is_grad_comm = isinstance(data, Gradient) and data.is_leaf_grad
        if (stree.collective_comm or is_grad_comm) and self.all_reduce_collective_comm(
                op, opcfg, wid, data, datacfg, stree.dev_topo, recompute=recompute):
            return
        if stree.collective_comm and self.reduce_scatter_collective_comm(
                op, opcfg, wid, data, datacfg, stree.dev_topo, recompute=recompute):
            return
        if stree.collective_comm and self.all_gather_write_collective_comm(
                op, opcfg, wid, data, datacfg, stree.dev_topo, recompute=recompute):
            return
        if stree.collective_comm and self.scatter_comm(op, opcfg, wid, data, datacfg, stree.dev_topo):
            return
        if stree.collective_comm and self.gather_comm(op, opcfg, wid, data, datacfg, stree.dev_topo):
            return

        for dst_id in range(datacfg.deg()):
            dst_interval = data.sub_tensors[dst_id]
            for src_id in range(opcfg.deg()):
                src_interval = op.sub_outs[src_id][wid]
                area = dst_interval.intersection(src_interval)
                if area <= 0:
                    continue

                rep_equal = len(data.sub_tasks[dst_id]) == len(op.sub_tasks[src_id])
                for i, dst_task in enumerate(data.sub_tasks[dst_id]):
                    # determine source task
                    if rep_equal:
                        src_task = op.sub_tasks[src_id][i]
                    else:
                        bw = -999
                        for src_task_c in op.sub_tasks[src_id]:
                            if stree.dev_topo.bw(src_task_c.dev_id,
                                                 dst_task.dev_id) > bw:
                                bw = stree.dev_topo.bw(src_task_c.dev_id,
                                                       dst_task.dev_id)
                                src_task = src_task_c
                    if dst_task.intermediate_data is not None:
                        if dst_task.intermediate_data.dev_id == src_task.dev_id:
                            if recompute:
                                src_task.recompute_task.add_next_task(dst_task.recompute_task.intermediate_data)
                            else:
                                src_task.add_next_task(dst_task.intermediate_data)
                            continue
                    if src_task.dev_id == dst_task.dev_id:
                        if recompute:
                            src_task.recompute_task.add_next_task(dst_task.recompute_task)
                        else:
                            src_task.add_next_task(dst_task)
                        continue

                    # produce intermediate tensor
                    inter_mem = MemTask(data,
                                        dst_id,
                                        stree.dev_topo.dev(src_task.dev_id),
                                        nelements=src_interval.nelements())
                    if recompute:
                        src_task.recompute_task.add_next_task(inter_mem)
                    else:
                        self.task_ids.add(inter_mem.id)
                        src_task.add_next_task(inter_mem)
                    src_task = inter_mem

                    volume = area * size_of_datatype(data.dtype) / 1e6
                    task = CommTask(volume,
                                    stree.dev_topo.dev(src_task.dev_id),
                                    stree.dev_topo.dev(dst_task.dev_id),
                                    is_grad_comm=is_grad_comm)
                    src_task.add_next_task(task)
                    if recompute:
                        task.add_next_task(dst_task.recompute_task)
                        if not self.is_optimizer:
                            dst_task.intermediate_data.recompute_task = inter_mem
                        dst_task.recompute_task.attached_recompute_task.update([
                            inter_mem.id, task.id
                        ])
                    else:
                        self.task_ids.add(task.id)
                        task.add_next_task(dst_task)
                        if not self.is_optimizer:
                            dst_task.intermediate_data = inter_mem

    def dep_analysis(self, graph, stree):
        for op_id, op in self.ops.items():
            opcfg = graph.op_config[op_id]
            for wid, data in enumerate(op.write):
                datacfg = graph.data_config[data.id]
                # prepare intermediate data
                # offload and recomputation needs intermediate saved data for later usage
                comm_cpu = self.check_inter_cpu_gpu(opcfg.dev_type,
                                                    datacfg.dev_type)
                if comm_cpu == 'gpu2cpu':
                    fake_tasks, fake_mapping, fake_mesh = self.gpu_to_cpu(
                        data, datacfg, stree.dev_topo)
                    # replace
                    real_mapping = datacfg.mapping
                    real_mesh = datacfg.mesh
                    real_tasks = data.sub_tasks
                    datacfg.mapping = fake_mapping
                    datacfg.mesh = fake_mesh
                    data.sub_tasks = fake_tasks
                elif comm_cpu == 'cpu2gpu':
                    raise NotImplementedError

                self.op_write_data(op, wid, data, opcfg, datacfg, stree)
                if op.recomputation:
                    self.op_write_data(op, wid, data, opcfg, datacfg, stree, recompute=True)

                if comm_cpu == 'gpu2cpu':
                    # restore
                    datacfg.mapping = real_mapping
                    datacfg.mesh = real_mesh
                    data.sub_tasks = real_tasks

        for op_id, op in self.ops.items():
            opcfg = graph.op_config[op_id]
            for rid, data in enumerate(op.read):
                datacfg = graph.data_config[data.id]
                # check whether need cpu2gpu or gpu2cpu comm
                comm_cpu = self.check_inter_cpu_gpu(datacfg.dev_type,
                                                    opcfg.dev_type)
                if comm_cpu == 'cpu2gpu':
                    fake_tasks, fake_mapping, fake_mesh = self.cpu_to_gpu(
                        data, datacfg, stree.dev_topo, op, rid)
                    # replace
                    real_mapping = datacfg.mapping
                    real_mesh = datacfg.mesh
                    real_tasks = data.sub_tasks
                    datacfg.mapping = fake_mapping
                    datacfg.mesh = fake_mesh
                    data.sub_tasks = fake_tasks
                elif comm_cpu == 'gpu2cpu':
                    raise NotImplementedError

                self.op_read_data(op, rid, data, opcfg, datacfg, stree, graph)
                if op.recomputation:
                    self.op_read_data(op, rid, data, opcfg, datacfg, stree, graph, recompute=True)

                if comm_cpu == 'cpu2gpu':
                    # restore
                    datacfg.mapping = real_mapping
                    datacfg.mesh = real_mesh
                    data.sub_tasks = real_tasks

        if not self.forward and not self.is_optimizer:
            bwop_map = {}
            for fwop_id, bwop_id in graph.fwop_map.items():
                bwop_map[bwop_id] = fwop_id
            last_op = None
            for op_id, op in sorted(self.ops.items()):
                fwop_id = bwop_map[op_id]
                if graph.ops[fwop_id].recomputation:
                    for _, sub_tasks in graph.ops[fwop_id].sub_tasks.items():
                        for tsk in sub_tasks:
                            self.task_ids.add(tsk.recompute_task.id)
                            for tid in tsk.recompute_task.attached_recompute_task:
                                self.task_ids.add(tid)
                    for data in graph.ops[fwop_id].write:
                        if data.sub_tasks[0][0].recompute_task is None:
                            continue
                        for _, sub_tasks in data.sub_tasks.items():
                            for tsk in sub_tasks:
                                self.task_ids.add(tsk.recompute_task.id)
                                for tid in tsk.recompute_task.attached_recompute_task:
                                    self.task_ids.add(tid)

                if graph.ops[fwop_id].recomputation and last_op is None:
                    last_op = graph.ops[fwop_id]
                if not graph.ops[fwop_id].recomputation and last_op is not None:
                    for _, src_tasks in op.sub_tasks.items():
                        for src_tsk in src_tasks:
                            for _, dst_tasks in last_op.sub_tasks.items():
                                for dst_tsk in dst_tasks:
                                    src_tsk.add_next_task(dst_tsk.recompute_task)
                    last_op = None

        for _, data in self.datas.items():
            for ctr_data in data.control:
                for _, src_tasks in ctr_data.sub_tasks.items():
                    for src_tsk in src_tasks:
                        for _, dst_tasks in data.sub_tasks.items():
                            for dst_tsk in dst_tasks:
                                src_tsk.add_next_task(dst_tsk)

        # optimization
        self.fuse_all_to_all()
        if not self.forward and not self.is_optimizer:
            self.bucket_gradient_allreduce(
                bucket_size=stree.bucket_size,
                overlap_grad_comm=stree.overlap_grad_comm)
        self.prefetch_control(graph)
        if self.is_optimizer:
            self.fuse_param_write()

    def _dep_analysis(self, graph, stree):
        assert False
        # >>>>> TB change
        for op_id, op in self.ops.items():
            opcfg = graph.op_config[op_id]
            recomputation = False
            if self.forward:
                for i, data in enumerate(op.write):
                    datacfg = graph.data_config[data.id]
                    if datacfg.is_recompute():
                        recomputation = True
                        break

            for i, data in enumerate(op.read):
                datacfg = graph.data_config[data.id]
                # check whether need cpu2gpu or gpu2cpu comm
                comm_cpu = self.check_inter_cpu_gpu(datacfg.dev_type,
                                                    opcfg.dev_type)
                if comm_cpu == 'cpu2gpu':
                    fake_tasks, fake_mapping, _ = self.cpu_to_gpu(
                        data, datacfg, stree.dev_topo, op, i)
                    # replace
                    real_mapping = datacfg.mapping
                    real_tasks = data.sub_tasks
                    datacfg.mapping = fake_mapping
                    data.sub_tasks = fake_tasks
                elif comm_cpu == 'gpu2cpu':
                    raise NotImplementedError
                # check whether is allgather
                if stree.collective_comm and self.all_gather_collective_comm(
                        op, opcfg, i, data, datacfg, stree.dev_topo):
                    if comm_cpu == 'cpu2gpu':
                        # recover
                        datacfg.mapping = real_mapping
                        data.sub_tasks = real_tasks
                    continue
                for dst_id in range(opcfg.deg()):
                    dst_interval = op.sub_ins[dst_id][i]
                    for src_id in range(datacfg.deg()):
                        src_interval = data.sub_tensors[src_id]

                        area = dst_interval.intersection(src_interval)
                        if area <= 0:
                            continue

                        if datacfg.mapping[src_id][0] == MapType.DROP:
                            for dst_task in op.sub_tasks[dst_id]:
                                src_task = data.sub_tasks[src_id][0]
                                if src_task.attach_dev != dst_task.dev_id:
                                    volume = area * size_of_datatype(
                                        data.dtype) / 1e6
                                    src_dev = stree.dev_topo.dev(
                                        src_task.attach_dev)
                                    dst_dev = stree.dev_topo.dev(
                                        dst_task.dev_id)
                                    task = CommTask(volume, src_dev, dst_dev)
                                    self.task_ids.add(task.id)
                                    # if op.depth == data.producer[0][0].depth + 1:
                                    if self.forward:
                                        src_task.add_next_task(task)
                                        task.add_next_task(dst_task)

                                        if recomputation:
                                            # recompute graph
                                            rtask = CommTask(
                                                volume, src_dev, dst_dev)
                                            rtask.is_recomputation = True
                                            src_task.recompute_task.add_next_task(
                                                rtask)
                                            rtask.add_next_task(
                                                dst_task.recompute_task)
                                            dst_task.attached_recompute_task.append(
                                                rtask)
                                    else:
                                        src_task.recompute_task.add_next_task(
                                            task)
                                        task.add_next_task(dst_task)
                                        self.add_recompute_dep(
                                            src_task.recompute_task)

                                else:
                                    if self.forward:
                                        src_task.add_next_task(dst_task)
                                        if recomputation:
                                            src_task.recompute_task.add_next_task(
                                                dst_task.recompute_task)
                                    else:
                                        src_task.recompute_task.add_next_task(
                                            dst_task)
                                        self.add_recompute_dep(
                                            src_task.recompute_task)
                            continue

                        for dst_task in op.sub_tasks[dst_id]:
                            bw = -999
                            for src_task_c in data.sub_tasks[src_id]:
                                if src_task_c.intermediate_data is not None and op.depth == data.producer[
                                        0][0].depth + 1:
                                    src_tc = src_task_c.intermediate_data
                                else:
                                    src_tc = src_task_c

                                if stree.dev_topo.bw(src_tc.dev_id,
                                                     dst_task.dev_id) > bw:
                                    bw = stree.dev_topo.bw(
                                        src_tc.dev_id, dst_task.dev_id)
                                    src_task = src_tc
                            if src_task.dev_id == dst_task.dev_id:
                                src_task.add_next_task(dst_task)
                                if recomputation:
                                    src_task.add_next_task(
                                        dst_task.recompute_task)
                                continue

                            volume = area * size_of_datatype(data.dtype) / 1e6
                            task = CommTask(
                                volume, stree.dev_topo.dev(src_task.dev_id),
                                stree.dev_topo.dev(dst_task.dev_id))
                            self.task_ids.add(task.id)

                            # make intemediate data
                            inter_mem = MemTask(
                                data, src_id,
                                stree.dev_topo.dev(src_task.dev_id))
                            self.task_ids.add(inter_mem.id)

                            src_task.add_next_task(task)
                            task.add_next_task(inter_mem)
                            inter_mem.add_next_task(dst_task)
                            if recomputation:
                                rtask = CommTask(
                                    volume,
                                    stree.dev_topo.dev(src_task.dev_id),
                                    stree.dev_topo.dev(dst_task.dev_id))
                                rtask.is_recomputation = True
                                src_task.add_next_task(rtask)
                                rtask.add_next_task(dst_task.recompute_task)
                                dst_task.attached_recompute_task.append(rtask)
                if comm_cpu == 'cpu2gpu':
                    # recover
                    datacfg.mapping = real_mapping
                    data.sub_tasks = real_tasks

            for i, data in enumerate(op.write):
                datacfg = graph.data_config[data.id]
                comm_cpu = self.check_inter_cpu_gpu(opcfg.dev_type,
                                                    datacfg.dev_type)
                if comm_cpu == 'gpu2cpu':
                    fake_tasks, fake_mapping, _ = self.gpu_to_cpu(
                        data, datacfg, stree.dev_topo)
                    # replace
                    real_mapping = datacfg.mapping
                    real_tasks = data.sub_tasks
                    datacfg.mapping = fake_mapping
                    data.sub_tasks = fake_tasks
                elif comm_cpu == 'cpu2gpu':
                    raise NotImplementedError

                if stree.collective_comm and self.all_reduce_collective_comm(
                        op, opcfg, i, data, datacfg, stree.dev_topo):
                    if comm_cpu == 'gpu2cpu':
                        # recover
                        datacfg.mapping = real_mapping
                        data.sub_tasks = real_tasks
                    continue
                if stree.collective_comm and self.reduce_scatter_collective_comm(
                        op, opcfg, i, data, datacfg, stree.dev_topo):
                    if comm_cpu == 'gpu2cpu':
                        # recover
                        datacfg.mapping = real_mapping
                        data.sub_tasks = real_tasks
                    continue
                if stree.collective_comm and self.all_gather_write_collective_comm(
                        op, opcfg, i, data, datacfg, stree.dev_topo):
                    if comm_cpu == 'gpu2cpu':
                        # recover
                        datacfg.mapping = real_mapping
                        data.sub_tasks = real_tasks
                    continue
                is_grad_comm = isinstance(data, Gradient) and data.is_leaf_grad

                for dst_id in range(datacfg.deg()):
                    dst_interval = data.sub_tensors[dst_id]

                    if datacfg.mapping[dst_id][0] == MapType.DROP:
                        assert datacfg.deg() == opcfg.deg()
                        assert op.sub_outs[dst_id][i] == dst_interval
                        src_task = op.sub_tasks[dst_id][0]
                        dst_task = data.sub_tasks[dst_id][0]
                        dst_task.attach_dev = src_task.dev_id
                        src_task.add_next_task(dst_task)

                        # build recompute graph
                        assert self.forward
                        dst_task.recompute_task.attach_dev = src_task.dev_id
                        src_task.recompute_task.add_next_task(
                            dst_task.recompute_task)
                        continue

                    for src_id in range(opcfg.deg()):
                        src_interval = op.sub_outs[src_id][i]
                        area = dst_interval.intersection(src_interval)
                        if area <= 0:
                            continue

                        for dst_task in data.sub_tasks[dst_id]:
                            # determine source task
                            bw = -999
                            for src_task_c in op.sub_tasks[src_id]:
                                if stree.dev_topo.bw(src_task_c.dev_id,
                                                     dst_task.dev_id) > bw:
                                    bw = stree.dev_topo.bw(
                                        src_task_c.dev_id, dst_task.dev_id)
                                    src_task = src_task_c
                            if src_task.dev_id == dst_task.dev_id:
                                src_task.add_next_task(dst_task)
                                continue

                            # produce intermediate tensor
                            inter_mem = MemTask(
                                data, dst_id,
                                stree.dev_topo.dev(src_task.dev_id))
                            self.task_ids.add(inter_mem.id)
                            dst_task.intermediate_data = inter_mem
                            src_task.add_next_task(inter_mem)
                            src_task = inter_mem

                            volume = area * size_of_datatype(data.dtype) / 1e6
                            task = CommTask(
                                volume,
                                stree.dev_topo.dev(src_task.dev_id),
                                stree.dev_topo.dev(dst_task.dev_id),
                                is_grad_comm=is_grad_comm)
                            self.task_ids.add(task.id)
                            src_task.add_next_task(task)
                            task.add_next_task(dst_task)
                            if data.id == 310:
                                print('Graph:', src_task, '-->', task, '-->',
                                      dst_task)
                if comm_cpu == 'gpu2cpu':
                    # recover
                    datacfg.mapping = real_mapping
                    data.sub_tasks = real_tasks

        for _, data in self.datas.items():
            for ctr_data in data.control:
                for _, src_tasks in ctr_data.sub_tasks.items():
                    for src_tsk in src_tasks:
                        for _, dst_tasks in data.sub_tasks.items():
                            for dst_tsk in dst_tasks:
                                src_tsk.add_next_task(dst_tsk)

        # optimization
        if not self.forward and not self.is_optimizer:
            self.bucket_gradient_allreduce(
                bucket_size=stree.bucket_size,
                overlap_grad_comm=stree.overlap_grad_comm)
        self.prefetch_control(graph)
        # >>>>> TB change

    def mark_cross_stage_comm(self):
        for tsk_id in self.task_ids:
            task = Task.get(tsk_id)
            if task.type == TaskType.TASK_COMM and task.comm_type == 'p2p':
                cur_src = task.src in self.dev_ids and task.dst not in self.dev_ids
                cur_dst = task.src not in self.dev_ids and task.dst in self.dev_ids
                if cur_src or cur_dst:
                    task.is_cross_stage = True

    def fuse_all_to_all(self):
        # mainly for DLRM
        scatter_group, gather_group = [], []
        for task_id in self.task_ids:
            task = Task.get(task_id)
            if task.type == TaskType.TASK_COMM:
                if task.comm_type == 'scatter':
                    scatter_group.append(task)
                elif task.comm_type == 'gather':
                    gather_group.append(task)

        def can_fuse(task_list):
            if len(task_list) <= 0:
                return False, None
            group, root_group, root_volume = None, [], {}
            for task in task_list:
                if group is None:
                    group = tuple(sorted(task.group))
                if tuple(sorted(task.group)) != group:
                    return False, None
                root = task.name.split('-')[1].split('_')[0]
                if root not in root_group:
                    root_group.append(root)
                    root_volume[root] = 0
                root_volume[root] += task.volume
            if tuple(sorted(root_group)) != group:
                return False, None
            return True, root_volume

        def fuse_tasks(task_lists, root_volume):
            volume = max(list(root_volume.values()))
            # volume = len(task_lists[0].group) * task_lists[0].volume
            all_to_all_task = CommTask(volume, None, None,
                                       collective='{}_alltoall'.format(task_lists[0].name),
                                       is_grad_comm=task_lists[0].is_grad_comm,
                                       comm_type='all_to_all',
                                       group=task_lists[0].group)
            self.task_ids.add(all_to_all_task.id)
            for task in task_lists:
                self.task_ids.remove(task.id)
                for prev_tk in task.prev_tasks:
                    prev_tk.remove_next_task(task)
                    prev_tk.add_next_task(all_to_all_task)
                for next_tk in task.next_tasks:
                    next_tk.remove_prev_task(task)
                    all_to_all_task.add_next_task(next_tk)

        fusable, root_volume = can_fuse(scatter_group)
        if fusable:
            fuse_tasks(scatter_group, root_volume)
        fusable, root_volume = can_fuse(gather_group)
        if fusable:
            fuse_tasks(gather_group, root_volume)

    def bucket_gradient_allreduce(self,
                                  bucket_size=25,
                                  overlap_grad_comm=True):
        ndeps = {}
        tasks_queue = queue.Queue()

        def get_deps_in_cur_graph(cur_task):
            cnt = 0
            for tsk in cur_task.prev_tasks:
                if tsk.id in self.task_ids:
                    cnt += 1
            return cnt

        for task_id in self.task_ids:
            task = Task.get(task_id)
            ndeps[task_id] = get_deps_in_cur_graph(task)

            if ndeps[task_id] == 0:
                task.depth = 0
                tasks_queue.put(task)

        all_reduce_tasks, reduce_scatter_tasks = [], []
        while not tasks_queue.empty():
            task = tasks_queue.get()
            for ntask in task.next_tasks:
                if ntask.id not in self.task_ids:
                    continue
                ntask.depth = max(ntask.depth, task.depth + 1)
                ndeps[ntask.id] -= 1
                if ndeps[ntask.id] == 0:
                    tasks_queue.put(ntask)
                    if ntask.type == TaskType.TASK_COMM and ntask.is_grad_comm:
                        if ntask.comm_type == 'all_reduce':
                            all_reduce_tasks.append(ntask)
                        elif ntask.comm_type == 'reduce_scatter':
                            reduce_scatter_tasks.append(ntask)

        def run_fuse(task_list):
            # sort tasks in occurance order during backward pass
            task_list = sorted(task_list, key=lambda tk: (tk.depth, -tk.id))

            # filter comm tasks of the same gradient
            commnicated, filtered_list = set(), []
            for task in reversed(task_list):
                devs = sorted([tsk.dev_id for tsk in task.next_tasks])
                grad_id = (task.next_tasks[0].data.id, task.next_tasks[0].index,
                           tuple(devs))
                if grad_id not in commnicated:
                    commnicated.add(grad_id)
                    filtered_list.append(task)
                else:
                    for prev_tk in task.prev_tasks:
                        prev_tk.remove_next_task(task)
                    for next_tk in task.next_tasks:
                        next_tk.remove_prev_task(task)
                    self.task_ids.remove(task.id)
            task_list = list(reversed(filtered_list))

            # build buckets
            if overlap_grad_comm:
                buckets = [kDefaultFirstBucketBytes, bucket_size]
            else:
                buckets = [bucket_size]
            fused_volume, bucket_idx = defaultdict(lambda: 0), 0
            tb_fused = defaultdict(list)

            def fuse_tasks(task_lists, volume):
                fused_task = CommTask.make_task_like(task_lists[0], volume=volume)
                self.task_ids.add(fused_task.id)
                for task in task_lists:
                    self.task_ids.remove(task.id)
                    for prev_tk in task.prev_tasks:
                        prev_tk.remove_next_task(task)
                        prev_tk.add_next_task(fused_task)
                    for next_tk in task.next_tasks:
                        next_tk.remove_prev_task(task)
                        fused_task.add_next_task(next_tk)

            for i, task in enumerate(task_list):
                key = tuple(sorted(task.group))
                tb_fused[key].append(task)
                fused_volume[key] += task.volume
                if fused_volume[key] > buckets[bucket_idx]:
                    fuse_tasks(tb_fused[key], fused_volume[key])
                    # next bucket
                    fused_volume[key] = 0
                    bucket_idx += 1
                    buckets.append(bucket_size)
                    tb_fused[key] = []
                if i >= len(task_list) - 1:
                    for key, task_lists in tb_fused.items():
                        if fused_volume[key] > 0:
                            fuse_tasks(task_lists, fused_volume[key])

        run_fuse(all_reduce_tasks)
        run_fuse(reduce_scatter_tasks)

    def fuse_param_write(self):
        all_gather_tasks = set()
        for tsk_id in self.task_ids:
            task = Task.get(tsk_id)
            if task.type == TaskType.TASK_COMM and task.comm_type == 'all_gather':
                all_gather_tasks.add(task)

        fused_volume = defaultdict(lambda: 0)
        tb_fused = defaultdict(list)

        def fuse_tasks(task_lists, volume):
            fused_task = CommTask.make_task_like(task_lists[0], volume=volume)
            self.task_ids.add(fused_task.id)
            for task in task_lists:
                self.task_ids.remove(task.id)
                for prev_tk in task.prev_tasks:
                    prev_tk.remove_next_task(task)
                    prev_tk.add_next_task(fused_task)
                    # when write param, here suppose it use continuous param
                    # hence, no need to use intermediate parameter buffer
                    prev_tk.memory = 0
                for next_tk in task.next_tasks:
                    next_tk.remove_prev_task(task)
                    fused_task.add_next_task(next_tk)

        for i, task in enumerate(all_gather_tasks):
            key = tuple(sorted(task.group))
            tb_fused[key].append(task)
            fused_volume[key] += task.volume
        for key, task_lists in tb_fused.items():
            if fused_volume[key] > 0:
                fuse_tasks(task_lists, fused_volume[key])

    def prefetch_control(self, graph):
        graph_ops = sorted(self.ops.values(), key=lambda op: op.depth)
        for i, op in enumerate(graph_ops):
            if i + 2 >= len(graph_ops):
                continue
            # gpu prefetch control
            control_op = graph_ops[i + 2]
            if len(control_op.prefetch_from_gpu) > 0:
                opcfg = graph.op_config[op.id]
                opcfg_map = list(opcfg.mesh.reshape(-1))
                replicate_deg = opcfg.mesh.shape[-1]
                for rid, prefetchs in control_op.prefetch_from_gpu.items():
                    for dev, tasks in prefetchs.items():
                        if dev in opcfg_map:
                            map_index = opcfg_map.index(dev)
                            deg_id = map_index // replicate_deg
                            rep_id = map_index % replicate_deg
                            src_task = op.sub_tasks[deg_id][rep_id]
                            for dst_task in tasks:
                                src_task.add_next_task(dst_task)
                        else:
                            src_tasks = []
                            for _, op_tasks in op.sub_tasks.items():
                                src_tasks.extend(op_tasks)
                            for src_task in src_tasks:
                                for dst_task in tasks:
                                    src_task.add_next_task(dst_task)

            if i + 3 >= len(graph_ops):
                continue
            # cpu prefetch control
            control_op = graph_ops[i + 3]
            if len(control_op.prefetch_from_cpu) > 0:
                opcfg = graph.op_config[op.id]
                opcfg_map = list(opcfg.mesh.reshape(-1))
                replicate_deg = opcfg.mesh.shape[-1]
                for rid, prefetchs in control_op.prefetch_from_cpu.items():
                    for dev, tasks in prefetchs.items():
                        if dev in opcfg_map:
                            map_index = opcfg_map.index(dev)
                            deg_id = map_index // replicate_deg
                            rep_id = map_index % replicate_deg
                            src_task = op.sub_tasks[deg_id][rep_id]
                            for dst_task in tasks:
                                src_task.add_next_task(dst_task)
                        else:
                            src_tasks = []
                            for _, op_tasks in op.sub_tasks.items():
                                src_tasks.extend(op_tasks)
                            for src_task in src_tasks:
                                for dst_task in tasks:
                                    src_task.add_next_task(dst_task)

    def __lt__(self, other):
        return self.id < other.id


class GraphTransformer:

    def __init__(self, module, cost_type='roofline'):
        self.module = module
        self.cost_type = cost_type

        self.is_first_stage = True
        for data in self.module.ins:
            if len(data.producer) != 0:
                self.is_first_stage = False

        self.op_ids = []
        self.data_ids = set()

        def dfs(node):
            if len(node.children) == 0:
                assert node.op.id not in self.op_ids
                self.op_ids.append(node.op.id)
                self.data_ids.update(node.data_config)
                return
            for _, child in node.children.items():
                dfs(child)

        dfs(self.module)

    def transform(self,
                  graph,
                  stree,
                  optimizer=False,
                  reprofile=False,
                  profile_iters=10):
        create_begin = time.perf_counter()  # timer

        # replace share params
        for param_id in graph.share_parameters:
            param = graph.datas[param_id]
            if optimizer:
                self.data_ids.add(param_id)
            if param_id not in self.data_ids and param not in self.module.ins and param not in self.module.outs:
                continue
            for share_p in param.share_params:
                as_in, as_out = False, False
                for op_id in self.op_ids:
                    op = graph.ops[op_id]
                    if share_p in op.read:
                        as_in = True
                        self.data_ids.add(share_p.id)
                    if share_p in op.write:
                        as_out = True
                        self.data_ids.add(share_p.id)
                if as_in:
                    ins = list(self.module.ins)
                    ins.remove(param)
                    ins.append(share_p)
                    self.module.ins = tuple(ins)
                if as_out:
                    outs = list(self.module.outs)
                    outs.remove(param)
                    outs.append(share_p)
                    self.module.outs = tuple(outs)
                if (as_in or as_out) and param_id in self.data_ids:
                    self.data_ids.remove(param_id)

        fw_graph = TaskGraph(self.is_first_stage,
                             self.cost_type,
                             reprofile=reprofile,
                             profile_iters=profile_iters,
                             optimizer=optimizer)
        if graph.train and not optimizer:
            bw_graph = TaskGraph(cost_type=self.cost_type,
                                 reprofile=reprofile,
                                 profile_iters=profile_iters,
                                 forward=False)

        for op_id in self.op_ids:
            fw_graph.ops[op_id] = graph.ops[op_id]
            if graph.train and not optimizer and op_id in graph.fwop_map:
                bwop_id = graph.fwop_map[op_id]
                bw_graph.ops[bwop_id] = graph.ops[bwop_id]
        for data_id in self.data_ids:
            fw_graph.datas[data_id] = graph.datas[data_id]
            if graph.train and not optimizer and graph.datas[data_id].grad:
                grad = graph.datas[data_id].grad
                bw_graph.datas[grad.id] = grad
        fw_graph.ins = self.module.ins
        fw_graph.outs = self.module.outs
        if graph.train and not optimizer:
            bw_graph.ins = tuple([d.grad for d in self.module.outs if d.grad])
            bw_graph.outs = tuple([d.grad for d in self.module.ins if d.grad])

        fw_graph.create_tasks(graph, stree)
        if graph.train and not optimizer:
            bw_graph.create_tasks(graph, stree)
        create_end = time.perf_counter()  # timer
        self.task_create_time = create_end - create_begin

        fw_graph.dep_analysis(graph, stree)
        if not optimizer:
            fw_graph.set_schedule(self.module)
        if self.module.dev_mesh['op'] is not None:
            fw_graph.dev_ids.update(self.module.dev_mesh['op'].reshape(-1))
            fw_graph.mark_cross_stage_comm()
        if graph.train and not optimizer:
            bw_graph.dep_analysis(graph, stree)
            bw_graph.set_schedule(self.module)
            bw_graph.dev_ids.update(self.module.dev_mesh['op'].reshape(-1))
            bw_graph.mark_cross_stage_comm()

        dep_analysis_end = time.perf_counter()
        self.dep_analysis_time = dep_analysis_end - create_end

        if graph.train and not optimizer:
            fw_graph.task_create_time = self.task_create_time / 2
            fw_graph.dep_analysis_time = self.dep_analysis_time / 2
            bw_graph.task_create_time = self.task_create_time / 2
            bw_graph.dep_analysis_time = self.dep_analysis_time / 2
            return fw_graph, bw_graph
        fw_graph.task_create_time = self.task_create_time
        fw_graph.dep_analysis_time = self.dep_analysis_time
        return (fw_graph, )


def run_transform(module,
                  graph,
                  strategy_tree,
                  cost_type='roofline',
                  optimizer=False,
                  reprofile=False,
                  profile_iters=10):
    return GraphTransformer(module, cost_type=cost_type).transform(
        graph,
        strategy_tree,
        optimizer=optimizer,
        reprofile=reprofile,
        profile_iters=profile_iters)
