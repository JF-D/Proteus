import bisect
import heapq
import time
import numpy as np
from collections import defaultdict, namedtuple
from scipy.optimize import linear_sum_assignment
from proteus.ir.tensor import Buffer, Gradient, Parameter
from proteus.passes.transform import Task, MemTask, CompTask, CommTask, run_transform
from proteus.type import TaskType
from proteus.utils import ListQueue
from .cost_model import OpCostModel
from .tracer import SimTracer

_CHANNELS = None
_EXECUTORS = None
_PROFILER = None


def get_channel():
    return _CHANNELS

def get_executors():
    return _EXECUTORS

DEBUG = False
def Dprint(string):
    if DEBUG:
        print(string)


def record_event(task, dev_id=None):
    # threads
    # 0: gpu, 1: cpu, 2: gpu2gpu, 3: cpu2gpu, 4: gpu2cpu
    if isinstance(task, CompTask):
        dev_type, rank = task.dev_id.split(':')
        thread = 0 if dev_type == 'gpu' else 1
        _PROFILER.add_trace_event(
            task.name.split('_')[0], '{}-{}'.format(task.name, task.state),
            int(rank), thread, [task.start, task.end])
    elif isinstance(task, CommTask):
        rank = int(dev_id.split(':')[1])
        if task.comm_type == 'p2p':
            _PROFILER.add_trace_event('{}-{}'.format(task.name,
                                                     task.state), task.name,
                                      rank, 2, [task.start, task.end])
            # _PROFILER.add_trace_event('{}-{}'.format(task.name,
            #                                          task.state), task.name,
            #                           task.dst, 2, [task.start, task.end])
        elif task.comm_type in ['cpu2gpu', 'gpu2cpu']:
            thread = 4 if task.comm_type == 'cpu2gpu' else 5
            _PROFILER.add_trace_event('{}-{}'.format(task.name,
                                                     task.state), task.name,
                                      rank, thread, [task.start, task.end])
        else:
            thread = 3 if task.is_grad_comm else 2
            _PROFILER.add_trace_event('{}-{}'.format(task.name,
                                                     task.state), task.name,
                                      rank, thread, [task.start, task.end])


def record_memory_event(dev_id, clock, memory):
    _PROFILER.add_memory_event(dev_id, clock, memory)


TreeTask = namedtuple('TreeTask', ('id', 'group', 'type'))


class Channels:

    def __init__(self, dev_topo, grad_bucket=25, share_bandwidth=True, overlap_factor=0, megatron=False, FlexFlow=False):
        self.dev_topo = dev_topo
        self.grad_bucket = grad_bucket
        self.share_bandwidth = share_bandwidth
        self.overlap_factor = overlap_factor
        self.megatron = megatron
        self.FlexFlow = FlexFlow

        self._channel = {}
        self.recv_channel = {}
        self.exe_slots = {}
        self.grad = ListQueue()
        for i in range(dev_topo.ndevs):
            for dev_type in ['gpu', 'cpu2gpu', 'gpu2cpu']:
                self._channel[f'{dev_type}:{i}'] = {
                    'feature_clock': 0,
                    'feature': ListQueue(comm=True),
                    'grad_clock': 0,
                    'grad': ListQueue()
                }
            self.exe_slots[i] = {'feature': [], 'grad': []}

    @property
    def channels(self):
        return self._channel

    def put(self, task: CommTask, force_feature=True, overlapped=True):
        root = 0
        sorted_group, dev_types = set(), set()
        for i, dev_id in enumerate(task.group):
            dev_type, rank = dev_id.split(':')
            sorted_group.add(int(rank))
            dev_types.add(dev_type)
            if i == 0:
                root = int(rank)
        sorted_group = sorted(list(sorted_group))
        if task.comm_type in ['p2p', 'reduce']:
            assert len(dev_types) == 1
            root = sorted_group.index(root)
        elif task.comm_type in ['cpu2gpu', 'gpu2cpu']:
            assert len(dev_types) == 2 and len(sorted_group) == 1
        elif task.comm_type in ['scatter', 'gather']:
            root = int(task.name.split('-')[1].split('_')[0].split(':')[-1])
        rets = OpCostModel.comm_cost(task.comm_type,
                                          task.volume,
                                          None,
                                          group=sorted_group,
                                          root=root,
                                          FlexFlow=self.FlexFlow)
        task.set_cost(*rets)
        if self.megatron and task.comm_type == 'reduce_scatter':
            # to match Megatron-LM, replace reduce_scatter with all_reduce
            task.volume = task.volume * 2
        if task.is_grad_comm and overlapped:
            task.overlap_factor = self.overlap_factor
            task.cost = task.cost * (1 + self.overlap_factor)

        if task.comm_type in ['cpu2gpu', 'gpu2cpu']:
            task.wait_n = 1
            dev_id = f'{task.comm_type}:{root}'
            if self._channel[dev_id]['feature'].empty():
                task.wait_n -= 1
            self._channel[dev_id]['feature'].put(task)
        elif task.is_cross_stage:
            # TBD: modify cross stage comm to adapt to more general cases
            task.wait_n = 1
            if self._channel[task.src]['feature'].empty():
                task.wait_n -= 1
            self._channel[task.src]['feature'].put(task)
        else:
            queue_type = 'grad' if task.is_grad_comm else 'feature'
            task.wait_n = len(task.group)
            for dev_id in task.group:
                if self._channel[dev_id][queue_type].empty():
                    task.wait_n -= 1
                self._channel[dev_id][queue_type].put(task)

    def done(self):
        for i, chnl in self._channel.items():
            if not chnl['feature'].empty() or not chnl['grad'].empty():
                return False
        return True

    def get_total_feature_cost(self, src, dst):
        ret = 0
        q = self._channel[src][dst]['feature']
        for i in range(q.size()):
            ret += q[i].cost
        return ret

    def get_total_grad_volume(self, src, dst):
        ret = 0
        q = self._channel[src][dst]['grad']
        for i in range(q.size()):
            ret += q[i].volume
        return ret

    def check_overlap(self, dev, start, end):
        while True:
            if len(self.exe_slots[dev]['grad']) == 0:
                break
            tsk = self.exe_slots[dev]['grad'][0]
            a, b = tsk.start, tsk.end
            if b <= start:
                self.exe_slots[dev]['grad'].pop(0)
                continue
            elif a >= end:
                break
            else:
                return True

    def share_bandwidth_comm(self, share_tasks, dummy_delay=False):
        # CommPathType:   0,     1,     2,     3,     4,     5,     6
        #               "LOC", "NVL", "NVB", "PIX", "PXB", "PHB", "SYS"
        share_factor = defaultdict(lambda: 1)
        n_gpu_per_node = OpCostModel.cluster.n_gpu_per_node

        node_tasks_dict = defaultdict(set)
        for task in share_tasks:
            for dev_id in task.group:
                node_id = int(dev_id.split(':')[-1]) // n_gpu_per_node
                node_tasks_dict[node_id].add(task)
        for node_id, node_tasks in node_tasks_dict.items():
            n_cross = 0
            for task in node_tasks:
                n_cross = n_cross + (1 if task.cross_node else 0)
            tree_tasks = []
            for task in node_tasks:
                share_factor[task.id] = max(share_factor[task.id], n_cross)
                if task.type_intra > 2:
                    tree_tasks.append(TreeTask(task.id, task.group, task.type_intra))

            def split_tree_task(tree_task, mid):
                left_group, right_group = [], []
                for dev_id in tree_task.group:
                    rank = int(dev_id.split(':')[-1])
                    if rank < mid:
                        left_group.append(dev_id)
                    else:
                        right_group.append(dev_id)
                left_task = TreeTask(tree_task.id, left_group, tree_task.type)
                right_task = TreeTask(tree_task.id, right_group, tree_task.type)
                return left_task, right_task

            def divide_and_share(level, tasks_group, mid, group_size):
                if level <= 2:
                    return
                left_tasks, right_tasks = [], []
                left_paths, right_paths = 0, 0

                left_group, right_group = [], []
                for task in tasks_group:
                    if len(task.group) <= 0:
                        continue
                    ltask, rtask = split_tree_task(task, mid)
                    left_group.append(ltask)
                    right_group.append(rtask)
                    if task.type > level:
                        # check whether cross current level or in one child side
                        if len(ltask.group) > 0:
                            left_paths += 1
                            left_tasks.append(task)
                        if len(rtask.group) > 0:
                            right_paths += 1
                            right_tasks.append(task)
                    elif task.type == level:
                        # cross current level
                        left_paths += 1
                        right_paths += 1
                        left_tasks.append(task)
                        right_tasks.append(task)
                for task in left_tasks:
                    share_factor[task.id] = max(share_factor[task.id], left_paths)
                for task in right_tasks:
                    share_factor[task.id] = max(share_factor[task.id], right_paths)

                if level == 5 and group_size <= 2:
                    next_level = 3
                else:
                    next_level = level - 1

                ng_sz = group_size // 2
                divide_and_share(next_level, left_group, mid - ng_sz, ng_sz)
                divide_and_share(next_level, right_group, mid + ng_sz, ng_sz)

            divide_and_share(6, tree_tasks, n_gpu_per_node // 2, n_gpu_per_node // 2)
        # set share bandwidth cost
        find_sys = False
        for task in share_tasks:
            if task.type_intra == 6 and not task.cross_node:
                find_sys = True
        max_bw, total_bw = 0, 0
        for task in share_tasks:
            if task.type_intra == 6 and not task.cross_node:
                if share_factor[task.id] > 1:
                    share_factor[task.id] = share_factor[task.id] / 2
            else:
                if dummy_delay and find_sys:
                    share_factor[task.id] = share_factor[task.id] / 2
            max_bw = max(max_bw, task.bw / share_factor[task.id])
            total_bw += task.bw / share_factor[task.id]
        # if len(share_tasks) > 0:
        #     max_bw = total_bw / len(share_tasks)

        if not dummy_delay:
            for task in share_tasks:
                # if task.bw > max_bw:
                #     real_bw = max_bw
                # else:
                #     real_bw = task.bw / share_factor[task.id]
                real_bw = max_bw
                task.set_share_bw_cost(real_bw)
                task.share_factor = task.bw / real_bw
        return share_factor

    def _execute(self):
        ntasks = []

        cross_stage = False
        # while not self.done():
        if not self.done():
            exe_chnls, idle_tic = [], defaultdict(lambda: 0)
            exe_grad_chnls, idle_grad_tic = [], defaultdict(lambda: 0)
            for chnl_id, chnl in self._channel.items():
                if not chnl['feature'].empty():
                    task = chnl['feature'][0]
                    if task.wait_n == 0:
                        idle_tic[task.id] = max(idle_tic[task.id], chnl['feature_clock'])
                        exe_chnls.append((chnl_id, chnl))
                        if chnl['feature'][0].is_cross_stage:
                            cross_stage = True
                if not chnl['grad'].empty():
                    task = chnl['grad'][0]
                    if task.wait_n == 0:
                        idle_grad_tic[task.id] = max(idle_grad_tic[task.id], chnl['grad_clock'])
                        exe_grad_chnls.append((chnl_id, chnl))


            execed_tasks = set()
            # force cross_stage comm not overlap with grad_comm
            if cross_stage:
                exe_grad_chnls = set()
            # check share bandwidth and launch comm tasks
            # first launch gradient comm tasks
            if self.share_bandwidth:
                tb_launch_tasks = set()
                for (chnl_id, chnl) in exe_grad_chnls:
                    tsk = chnl['grad'][0]
                    tb_launch_tasks.add(tsk)
                self.share_bandwidth_comm(tb_launch_tasks)

            for (chnl_id, chnl) in exe_grad_chnls:
                tsk = chnl['grad'].get()
                if not chnl['grad'].empty():
                    chnl['grad'][0].wait_n -= 1

                if tsk.done:
                    self._channel[chnl_id]['grad_clock'] = tsk.end
                else:
                    self._channel[chnl_id]['grad_clock'] = tsk.set_start_time(
                        max(tsk.ready[tsk.state], idle_grad_tic[tsk.id]))

                if chnl_id.startswith('gpu:'):
                    dev_id = int(chnl_id.split(':')[1])
                    self.exe_slots[dev_id]['grad'].append(tsk)

                execed_tasks.add(tsk)

                if _PROFILER:
                    record_event(tsk, chnl_id)

            # second launch feature tasks
            # check if there is overlap between feature tasks and grad comm tasks
            if self.share_bandwidth:
                overlap_time = -1
                tb_launch_tasks, grad_tasks = set(), set()
                max_feature_bw, max_grad_bw = 0, 0
                for (chnl_id, chnl) in exe_chnls:
                    tsk = chnl['feature'][0]
                    tb_launch_tasks.add(tsk)
                    max_feature_bw = max(max_feature_bw, tsk.bw)

                    start = max(tsk.ready[tsk.state], idle_tic[tsk.id])
                    end = start + tsk.cost
                    dev_id = int(chnl_id.split(':')[1])
                    for grad_tsk in self.exe_slots[dev_id]['grad']:
                        if grad_tsk.end <= start:
                            continue
                        elif grad_tsk.start >= end:
                            break
                        else:
                            grad_tasks.add(grad_tsk)
                            max_grad_bw = max(max_grad_bw, grad_tsk.bw)
                            overlap_time = min(end, grad_tsk.end) - max(start, grad_tsk.start)
                # check share bandwidth between feature tasks
                self.share_bandwidth_comm(tb_launch_tasks)
                if overlap_time > 0:
                    # delay shared feature tasks and grad tasks
                    need_to_adjust = []
                    tb_launch_tasks.update(grad_tasks)
                    share_factors = self.share_bandwidth_comm(tb_launch_tasks, dummy_delay=True)
                    for tsk in tb_launch_tasks:
                        if tsk.is_grad_comm:
                            max_grad_bw = max(max_grad_bw, tsk.bw / share_factors[tsk.id])
                        else:
                            max_feature_bw = max(max_feature_bw, tsk.bw / share_factors[tsk.id])

                    for tsk in tb_launch_tasks:
                        if tsk.is_grad_comm:
                            if tsk.bw > max_grad_bw:
                                real_bw = max_grad_bw
                            else:
                                real_bw = tsk.bw / share_factors[tsk.id]
                        else:
                            if tsk.bw > max_feature_bw:
                                real_bw = max_feature_bw
                            else:
                                real_bw = tsk.bw / share_factors[tsk.id]
                        share_factors[tsk.id] = tsk.bw / real_bw
                        delay = overlap_time * (share_factors[tsk.id] / tsk.share_factor - 1)
                        # assert delay >= 0
                        delay = max(0, delay)
                        if delay > 0:
                            tsk.cost += delay
                            if tsk.is_grad_comm:
                                tsk.set_start_time(tsk.start)
                                need_to_adjust.append(tsk)
                                for chnl_id in tsk.group:
                                    dev_id = int(chnl_id.split(':')[1])
                                    idx = self.exe_slots[dev_id]['grad'].index(tsk)
                                    prev_task = tsk
                                    for hist_tsk in self.exe_slots[dev_id]['grad'][idx + 1:]:
                                        if hist_tsk.start >= prev_task.end:
                                            break
                                        hist_tsk.set_start_time(prev_task.end + 0.0001)
                                        prev_task = hist_tsk
                                        need_to_adjust.append(hist_tsk)
                    for tsk in need_to_adjust:
                        for ntsk in tsk.next_tasks:
                            ntsk.ready[tsk.state] = max(tsk.end, ntsk.ready[tsk.state])

            for (chnl_id, chnl) in exe_chnls:
                tsk = chnl['feature'].get()
                if not chnl['feature'].empty():
                    chnl['feature'][0].wait_n -= 1

                if tsk.done:
                    self._channel[chnl_id]['feature_clock'] = tsk.end
                else:
                    self._channel[chnl_id]['feature_clock'] = tsk.set_start_time(
                        max(tsk.ready[tsk.state], idle_tic[tsk.id]))

                if chnl_id.startswith('gpu:'):
                    dev_id = int(chnl_id.split(':')[1])
                    self.exe_slots[dev_id]['feature'].append(tsk)

                execed_tasks.add(tsk)

                if _PROFILER:
                    record_event(tsk, chnl_id)
                if cross_stage:
                    self._channel[chnl_id]['grad_clock'] = max(self._channel[chnl_id]['feature_clock'],
                                    self._channel[chnl_id]['grad_clock'])

            for tsk in execed_tasks:
                for ntsk in tsk.next_tasks:
                    if ntsk.type == TaskType.TASK_MEM:
                        get_executors()[ntsk.rank].pre_alloc(ntsk)

                for release_func in tsk.release_hooks[tsk.state]:
                    release_func(tsk.end, tsk.state)

                Dprint(f'[*] {tsk}')
                for ntsk in tsk.next_tasks:
                    ntsk.decrease_ndeps(tsk.state)
                    ntsk.ready[tsk.state] = max(tsk.end, ntsk.ready[tsk.state])
                    Dprint(f'    {ntsk}, {ntsk.dep_free()}, {ntsk.prev_tasks}')
                    if ntsk.dep_free():
                        ntasks.append(ntsk)
        return ntasks

    def execute(self, clear_grad=False):
        ntasks = []
        ret = self._execute()
        ntasks.extend(ret)
        return ntasks


class MemChunk:

    def __init__(self, task, ref_count):
        self.id = task.id
        self.name = task.name
        self.ref_count = ref_count
        self.release_func = None

    def release(self, clock, state):
        self.ref_count -= 1
        if self.ref_count <= 0:
            if self.release_func:
                self.release_func(clock, state)

    def __repr__(self):
        return self.name


class DevExecutor:

    def __init__(self, dev, overlap_factor=0):
        self.dev = dev
        self.overlap_factor = overlap_factor

        self.comp_queue = ListQueue()
        self.memory_chunks = {}

        self.max_mem_alloc = 0
        self.mem_alloc = 0
        self.clock = 0

    @property
    def id(self):
        return self.dev.id

    @property
    def type(self):
        return self.dev.type

    @property
    def memory(self):
        return self.dev.memory

    def pre_alloc(self, task):
        req_alloc = not (task.permanent and not task.first_round)
        if req_alloc:
            delta_memory = task.memory
            mem_chunk = MemChunk(task, len(task.next_tasks))

            if _PROFILER:
                record_memory_event(self.id, self.clock, delta_memory)

            def release_func(time, state):
                self.mem_alloc -= task.memory
                self.memory_chunks.pop((task.id, state))

                if _PROFILER:
                    record_memory_event(self.id, time, -task.memory)

            mem_chunk.release_func = release_func
            self.memory_chunks[(task.id, task.state)] = mem_chunk
            self.mem_alloc += delta_memory
            self.max_mem_alloc = max(self.mem_alloc, self.max_mem_alloc)

            if not hasattr(task, 'pre_allocated'):
                task.pre_allocated = set()
            task.pre_allocated.add(task.state)

    def alloc(self, task):
        task.set_start_time(task.ready[task.state])

        req_alloc = not (task.permanent and not task.first_round)
        task.first_round = False

        if req_alloc:
            delta_memory = 0

            if task.is_grad and (task.id,
                                 task.state - 1) in self.memory_chunks:
                self.memory_chunks.pop((task.id, task.state - 1))
                delta_memory -= task.memory

            if not hasattr(task, 'pre_allocated') or task.state not in task.pre_allocated:
                delta_memory += task.memory
                mem_chunk = MemChunk(task, len(task.next_tasks))

                if _PROFILER:
                    record_memory_event(self.id, self.clock, delta_memory)

                def release_func(time, state):
                    self.mem_alloc -= task.memory
                    self.memory_chunks.pop((task.id, state))

                    if _PROFILER:
                        record_memory_event(self.id, time, -task.memory)

                mem_chunk.release_func = release_func
                self.memory_chunks[(task.id, task.state)] = mem_chunk
            else:
                mem_chunk = self.memory_chunks[(task.id, task.state)]

            self.mem_alloc += delta_memory
            self.max_mem_alloc = max(self.mem_alloc, self.max_mem_alloc)
        else:
            if task.persist and (task.id, 1) not in self.memory_chunks:
                mem_chunk = self.memory_chunks[(task.id, task.state)]
            else:
                mem_chunk = self.memory_chunks[(task.id, 1)]

        if self.mem_alloc > self.memory:
            # print(f'out of memory: {self.mem_alloc}')
            pass
            # assert False

        if len(task.next_tasks) == 0:
            mem_chunk.release(self.clock, task.state)
        ntasks = []
        Dprint(f'[#] {task}')
        for ntsk in task.next_tasks:
            if not task.permanent:
                ntsk.release_hooks[task.state].append(mem_chunk.release)
            ntsk.decrease_ndeps(task.state)
            ntsk.ready[task.state] = max(task.end, ntsk.ready[task.state])
            Dprint(f'    {ntsk}, {ntsk.dep_free()}, {ntsk.prev_tasks}')
            if ntsk.dep_free():
                ntasks.append(ntsk)
        return ntasks

    def execute(self, sync_clock=False):
        if sync_clock:
            for stream in ['gpu', 'cpu2gpu', 'gpu2cpu']:
                clock1 = get_channel()._channel[f'{stream}:{self.id}']['feature_clock']
                clock2 = get_channel()._channel[f'{stream}:{self.id}']['grad_clock']
                self.clock = max(self.clock, clock1, clock2)
        ntasks = []
        # while not self.comp_queue.empty():
        if not self.comp_queue.empty():
            task = self.comp_queue.get()

            for ntsk in task.next_tasks:
                if ntsk.type == TaskType.TASK_MEM:
                    self.pre_alloc(ntsk)

            start = max(task.ready[task.state], self.clock) + 2e-3
            status = get_channel().check_overlap(self.id, start,
                                                 start + task.cost)
            if status:
                old_cost = task.cost
                task.cost = task.cost * (1 + self.overlap_factor)
            self.clock = task.set_start_time(start)
            if status:
                task.cost = old_cost

            if _PROFILER:
                record_event(task)

            for release_func in task.release_hooks[task.state]:
                release_func(self.clock, task.state)

            Dprint(f'[>] {task} ')
            for ntsk in task.next_tasks:
                ntsk.decrease_ndeps(task.state)
                ntsk.ready[task.state] = max(self.clock,
                                             ntsk.ready[task.state])
                Dprint(f'    {ntsk}, {ntsk.dep_free()}, {ntsk.prev_tasks}')
                if ntsk.dep_free():
                    ntasks.append(ntsk)
        return ntasks


class DevGroupDispatcher:

    def __init__(self, executors, interleave_freq=1, optimizer_overlap=False):
        self.executors = {ec.id: ec for ec in executors}
        self.fw_nodes = []
        self.bw_nodes = []
        self.count = {}

        self.cur_type = 'fw'
        self.cur_stage = 0
        self.prev_fw_stage = -1
        self.prev_bw_stage = -1
        self.interleave_freq = interleave_freq

        self.optimizer_overlap = optimizer_overlap

    def add_fw_node(self, node):
        self.fw_nodes.append(node)
        self.fw_nodes = sorted(self.fw_nodes)
        self.count[node.id] = 0

    def add_bw_node(self, node):
        self.bw_nodes.append(node)
        self.bw_nodes = sorted(self.bw_nodes)
        self.count[node.id] = 0

    def done(self):
        for node in self.fw_nodes:
            if not node.done():
                return False
        for node in self.bw_nodes:
            if not node.done():
                return False
        return True

    def reset(self):
        for node in self.fw_nodes:
            node.reset()
        for node in self.bw_nodes:
            node.reset()

    def interleave(self):
        if self.cur_type == 'fw':
            cur_stage = self.fw_nodes[self.cur_stage]
        else:
            cur_stage = self.bw_nodes[self.cur_stage]

        il_freq = self.interleave_freq
        if self.count[cur_stage.id] < il_freq and cur_stage.dep_free():
            if self.cur_type == 'fw':
                self.pre_fw_stage = self.cur_stage
            else:
                self.pre_bw_stage = self.cur_stage
            return cur_stage

        if self.cur_type == 'fw':
            for i in range(len(self.bw_nodes)):
                if self.bw_nodes[i].dep_free():
                    self.pre_fw_stage = self.cur_stage
                    self.pre_bw_stage = i
                    self.cur_stage = i
                    self.cur_type = 'bw'
                    self.count[cur_stage.id] = 0
                    return self.bw_nodes[i]
            for i in range(self.cur_stage + 1, len(self.fw_nodes)):
                if self.fw_nodes[i].dep_free():
                    self.cur_stage = i
                    self.prev_fw_stage = self.cur_stage
                    self.count[cur_stage.id] = 0
                    return self.fw_nodes[i]
            for i in range(self.cur_stage + 1):
                if self.fw_nodes[i].dep_free():
                    self.cur_stage = i
                    self.prev_fw_stage = self.cur_stage
                    self.count[cur_stage.id] = 0
                    return self.fw_nodes[i]
        else:
            for i in range(self.prev_fw_stage + 1, len(self.fw_nodes)):
                if self.fw_nodes[i].dep_free():
                    self.prev_bw_stage = self.cur_stage
                    self.cur_stage = i
                    self.prev_fw_stage = i
                    self.cur_type = 'fw'
                    self.count[cur_stage.id] = 0
                    return self.fw_nodes[i]
            for i in range(self.prev_fw_stage + 1):
                if self.fw_nodes[i].dep_free():
                    self.prev_bw_stage = self.cur_stage
                    self.cur_stage = i
                    self.prev_fw_stage = i
                    self.cur_type = 'fw'
                    self.count[cur_stage.id] = 0
                    return self.fw_nodes[i]
            for i in range(len(self.bw_nodes)):
                if self.bw_nodes[i].dep_free():
                    self.cur_stage = i
                    self.prev_bw_stage = i
                    self.count[cur_stage.id] = 0
                    return self.bw_nodes[i]
        return None

    def execute(self):
        cur_stage = self.interleave()
        if cur_stage:
            # init stage input dep
            init_tasks, init_tids = [], set()
            for data in cur_stage.ins:
                for _, sub_tasks in data.sub_tasks.items():
                    for tsk in sub_tasks:
                        if tsk.dep_free() and not tsk.done and tsk.rank in self.executors:
                            self.executors[tsk.rank].alloc(tsk)
                        for ntsk in tsk.next_tasks:
                            if ntsk.dep_free() and ntsk.id not in init_tids:
                                init_tasks.append(ntsk)
                                init_tids.add(ntsk.id)

            # init stage comm dep
            for tid in reversed(list(cur_stage.task_ids)):
                task = Task.get(tid)
                if not isinstance(Task.get(tid), CommTask):
                    if task.dep_free(
                    ) and task.is_recomputation and task.id not in init_tids:
                        init_tasks.append(task)
                        init_tids.add(task.id)
                    continue
                if task.dep_free(
                ) and not task.done and task.id not in init_tids:
                    init_tasks.append(Task.get(tid))
                    init_tids.add(task.id)

            if self.optimizer_overlap:
                init_tasks = sorted(init_tasks, key=lambda x: x.get_ready())

            for tsk in init_tasks:
                if tsk.id not in cur_stage.task_ids:
                    # deal with dep of cross stage comm
                    for ntsk in tsk.next_tasks:
                        if ntsk.dep_free() and ntsk.id in cur_stage.task_ids:
                            init_tasks.append(ntsk)
                            init_tids.add(ntsk.id)
                    continue
                if isinstance(tsk, CompTask):
                    self.executors[tsk.rank].comp_queue.put(tsk)
                elif isinstance(tsk, MemTask):
                    ntasks = self.executors[tsk.rank].alloc(tsk)
                    for task in ntasks:
                        if task.id not in init_tids:
                            init_tasks.append(task)
                            init_tids.add(task.id)
                else:
                    get_channel().put(tsk)

            self.count[cur_stage.id] += 1
            self.dev_execute(cur_stage)
        return cur_stage

    def dev_execute(self, tgraph):
        while not tgraph.macro_batch_done():
            ntasks = []
            if tgraph.is_optimizer:
                ret = get_channel().execute(True)
                ntasks.extend(ret)
            for k, executor in self.executors.items():
                ret = executor.execute(sync_clock=tgraph.is_optimizer
                                       and not self.optimizer_overlap)
                ntasks.extend(ret)
            no_overlap = tgraph.grad_not_done_only()
            for tsk in ntasks:
                if tsk.id not in tgraph.task_ids:
                    continue
                if isinstance(tsk, MemTask):
                    ntsks = self.executors[tsk.rank].alloc(tsk)
                    ntasks.extend(ntsks)
                elif isinstance(tsk, CompTask):
                    self.executors[tsk.rank].comp_queue.put(tsk)
                else:
                    get_channel().put(tsk, overlapped=not no_overlap)
            ntasks = get_channel().execute(no_overlap or tgraph.is_optimizer)
            for tsk in ntasks:
                if tsk.id not in tgraph.task_ids:
                    continue
                if isinstance(tsk, MemTask):
                    # print(self.executors)
                    ntsks = self.executors[tsk.rank].alloc(tsk)
                    ntasks.extend(ntsks)
                elif isinstance(tsk, CompTask):
                    self.executors[tsk.rank].comp_queue.put(tsk)
                else:
                    get_channel().put(tsk, overlapped=not no_overlap)
        # set schedule statistics
        tgraph.execute()


class Simulator:

    def __init__(self,
                 graph,
                 strategy_tree,
                 cost_type='roofline',
                 reprofile=False,
                 profile_iters=10,
                 optimizer_overlap=True,
                 cprofile_compile=False,
                 torch_profile=False,
                 share_bandwidth=True,
                 overlap_factor=0,
                 megatron=False,
                 FlexFlow=False,
                 cache_filename='cache'):
        self.graph = graph
        self.strategy_tree = strategy_tree
        self._cost_model = OpCostModel(cache_filename)
        self.flexflow = False #FlexFlow

        self.executors = {}
        for dev_id in range(self.strategy_tree.dev_topo.ndevs):
            self.executors[dev_id] = DevExecutor(
                self.strategy_tree.dev_topo.dev(dev_id),
                overlap_factor=overlap_factor)
        self.channels = Channels(self.strategy_tree.dev_topo,
                                 grad_bucket=strategy_tree.bucket_size,
                                 share_bandwidth=share_bandwidth,
                                 overlap_factor=overlap_factor,
                                 megatron=megatron,
                                 FlexFlow=FlexFlow)

        global _CHANNELS, _EXECUTORS
        _CHANNELS = self.channels
        _EXECUTORS = self.executors

        if torch_profile:
            import torch
            with torch.profiler.profile(
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        './log/sim_prof'),
                    record_shapes=False,
                    profile_memory=False,
            ) as prof:
                self.compile(self.graph, self.strategy_tree, cost_type,
                             reprofile, profile_iters, optimizer_overlap,
                             cprofile_compile)
                prof.step()
            print(prof.key_averages().table(sort_by='self_cuda_time_total'))
        else:
            self.compile(self.graph, self.strategy_tree, cost_type, reprofile,
                         profile_iters, optimizer_overlap, cprofile_compile)

    def compile(self,
                graph,
                strategy_tree,
                cost_type='roofline',
                reprofile=False,
                profile_iters=10,
                optimizer_overlap=True,
                cprofile_compile=False):
        if cprofile_compile:
            import cProfile, pstats, io
            from pstats import SortKey
            pr = cProfile.Profile()
            pr.enable()
        compile_begin = time.perf_counter()  # timer

        dev_groups = set()
        task_graphs = []
        if strategy_tree.root.pp_splited:
            for name, vnode in strategy_tree.root.children.items():
                task_graph = run_transform(vnode,
                                           graph,
                                           strategy_tree,
                                           cost_type=cost_type,
                                           reprofile=reprofile,
                                           profile_iters=profile_iters)
                task_graphs.append(list(task_graph))

                dev_groups.add(
                    tuple(set(sorted(vnode.dev_mesh['op'].reshape(-1)))))
        else:
            task_graph = run_transform(strategy_tree.root,
                                       graph,
                                       strategy_tree,
                                       cost_type=cost_type,
                                       reprofile=reprofile,
                                       profile_iters=profile_iters)
            task_graphs.append(list(task_graph))
            dev_groups.add(
                tuple(
                    set(sorted(
                        strategy_tree.root.dev_mesh['op'].reshape(-1)))))

        # set stage dep
        for i in range(len(task_graphs) - 1):
            task_graphs[i][0].add_next(task_graphs[i + 1][0])
            task_graphs[i][0].set_stage_id(i)
        task_graphs[len(task_graphs) - 1][0].set_stage_id(len(task_graphs) - 1)
        if graph.train:
            task_graphs[len(task_graphs) - 1][0].add_next(
                task_graphs[len(task_graphs) - 1][1])
            task_graphs[len(task_graphs) - 1][1].set_stage_id(len(task_graphs))
            for i in range(len(task_graphs) - 1):
                task_graphs[i + 1][1].add_next(task_graphs[i][1])
                task_graphs[i][1].set_stage_id(2 * len(task_graphs) - 1 - i)

            for i in range(len(task_graphs)):
                task_graphs[i][1].set_forward(task_graphs[i][0])

        # dev group dispatcher
        self.dev_group_dis = {}
        for dev_group in dev_groups:
            key = []
            
            for dev_id in dev_group:
                key.append(int(dev_id.split(':')[1]))
            # print(key)
            self.dev_group_dis[tuple(sorted(key))] = DevGroupDispatcher(
                [self.executors[dev_id] for dev_id in key],
                self.strategy_tree.root.pconfig['schedule']['interleave_freq'])

        # deal with cross stage comm
        all_stages = [tg for task_graph in task_graphs for tg in task_graph]
        for task_graph in task_graphs:
            for tg in task_graph:
                ctasks = []
                for data in tg.outs:
                    for _, sub_tasks in data.sub_tasks.items():
                        for tsk in sub_tasks:
                            for ntsk in tsk.next_tasks:
                                if ntsk.id not in tg.task_ids and isinstance(
                                        ntsk, CommTask):
                                    ctasks.append(ntsk)

                # move cross stage tasks to prev_stage
                for tsk in ctasks:
                    for graphs in task_graphs:
                        for gh in graphs:
                            if tsk.id in gh.task_ids:
                                gh.task_ids.remove(tsk.id)
                    tg.task_ids.add(tsk.id)

        for tg in all_stages:
            for data in tg.ins:
                if len(data.producer) == 0:
                    for _, sub_tasks in data.sub_tasks.items():
                        for tsk in sub_tasks:
                            if tsk.id in tg.task_ids and tsk.dev_id not in tg.dev_ids:
                                tg.task_ids.remove(tsk.id)
                                for gh in all_stages:
                                    if len(tsk.next_tasks) > 0 and tsk.next_tasks[0].id in gh.task_ids:
                                        gh.task_ids.add(tsk.id)

            for data in tg.outs:
                datacfg = graph.data_config[data.id]
                if datacfg.mapping[0][1] in tg.dev_ids:
                    continue
                for gh in all_stages:
                    if gh.id != tg.id and data in gh.ins:
                        for _, sub_tasks in data.sub_tasks.items():
                            for tsk in sub_tasks:
                                if tsk.id in tg.task_ids:
                                    gh.task_ids.add(tsk.id)
                                    tg.task_ids.remove(tsk.id)
                                else:
                                    assert tsk.id in gh.task_ids

        for task_graph in task_graphs:
            for i, tg in enumerate(task_graph):
                key = set()
                for dev_id in sorted(list(tg.dev_ids)):
                    key.add(int(dev_id.split(':')[1]))
                key = tuple(sorted(key))
                if i == 0:
                    self.dev_group_dis[key].add_fw_node(tg)
                else:
                    self.dev_group_dis[key].add_bw_node(tg)
                for task_id in tg.task_ids:
                    task = Task.get(task_id)
                    if isinstance(task, CommTask) and task.is_grad_comm:
                        task.set_force_once()
                    nbatch = self.strategy_tree.root.pconfig['schedule'][
                        'n_macro_batch']
                    task.set_n_max_state(nbatch)

        for k in self.dev_group_dis:
            self.dev_group_dis[k].reset()

        self.optimizer = run_transform(strategy_tree.optimizer,
                                       graph,
                                       strategy_tree,
                                       cost_type=cost_type,
                                       optimizer=True,
                                       reprofile=reprofile,
                                       profile_iters=profile_iters)[0]
        self.optimizer.reset(
            self.strategy_tree.root.pconfig['schedule']['n_macro_batch'])
        self.optimizer_dev_group = DevGroupDispatcher(
            list(self.executors.values()), optimizer_overlap=optimizer_overlap)
        self.optimizer_dev_group.add_fw_node(self.optimizer)

        compile_end = time.perf_counter()  # timer
        self.compile_time = compile_end - compile_begin
        if cprofile_compile:
            pr.disable()
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())

        self.task_create_time = 0
        self.dep_analysis_time = 0
        for task_graph in task_graphs:
            for tg in task_graph:
                self.task_create_time += tg.task_create_time
                self.dep_analysis_time += tg.dep_analysis_time
        self.task_create_time += self.optimizer.task_create_time
        self.dep_analysis_time += self.optimizer.dep_analysis_time

    def done(self):
        for _, dev_group in self.dev_group_dis.items():
            if not dev_group.done():
                return False
        return True

    def run(self, profile=None, cprofile_analysis=False):
        if profile and not cprofile_analysis:
            global _PROFILER
            _PROFILER = SimTracer(self.strategy_tree.dev_topo.ndevs)

        if cprofile_analysis:
            import cProfile, pstats, io
            from pstats import SortKey
            pr = cProfile.Profile()
            pr.enable()
        run_begin = time.perf_counter()

        if self.flexflow:
            # sim flexflow
            return self.run_flexflow()

        # initialize
        for _, data in self.graph.datas.items():
            if len(data.producer) == 0:
                for _, sub_tasks in data.sub_tasks.items():
                    for task in sub_tasks:
                        task.ready[task.state] = 0
                        self.executors[task.rank].alloc(task)

        while not self.done():
            stages = []
            for _, dev_group in self.dev_group_dis.items():
                cur_stage = dev_group.execute()
                stages.append(cur_stage.id if cur_stage else cur_stage)

            for _, dev_group in self.dev_group_dis.items():
                for node in dev_group.fw_nodes:
                    for callback in node.callbacks:
                        callback()
                    node.callbacks = []
                for node in dev_group.bw_nodes:
                    for callback in node.callbacks:
                        callback()
                    node.callbacks = []

        self.optimizer.max_available_batch = 1
        self.optimizer_dev_group.execute()

        cost = 0
        for i, executor in self.executors.items():
            cost = max(cost, executor.clock)
        for i, chnl in get_channel().channels.items():
            cost = max(cost, chnl['feature_clock'], chnl['grad_clock'])
        self.iter_speed = cost

        run_end = time.perf_counter()
        self.run_time = run_end - run_begin
        if cprofile_analysis:
            pr.disable()
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())

        if profile and not cprofile_analysis:
            _PROFILER.export(profile)

        self._cost_model.save()
        return cost

    def run_flexflow(self):
        ready_queue = []

        for _, data in self.graph.datas.items():
            if len(data.producer) == 0:
                for _, sub_tasks in data.sub_tasks.items():
                    for task in sub_tasks:
                        task.ready[task.state] = 0
                        heapq.heappush(ready_queue, (0, task.id, task))

        devices = {dev_id: 0 for dev_id in range(self.strategy_tree.dev_topo.ndevs)}
        while len(ready_queue) > 0:
            _, _, cur_task = heapq.heappop(ready_queue)

            if cur_task.type in [TaskType.TASK_COMP, TaskType.TASK_MEM]:
                dev_id = cur_task.rank
                ready_t = max(cur_task.get_ready(), devices[dev_id])
                end_t = cur_task.set_start_time(ready_t)
                devices[dev_id] = end_t
            elif cur_task.type == TaskType.TASK_COMM:
                root = 0
                sorted_group, dev_types = set(), set()
                for i, dev_id in enumerate(cur_task.group):
                    dev_type, rank = dev_id.split(':')
                    sorted_group.add(int(rank))
                    dev_types.add(dev_type)
                    if i == 0:
                        root = int(rank)
                sorted_group = sorted(list(sorted_group))
                if cur_task.comm_type in ['p2p', 'reduce']:
                    assert len(dev_types) == 1
                    root = sorted_group.index(root)
                elif cur_task.comm_type in ['cpu2gpu', 'gpu2cpu']:
                    assert len(dev_types) == 2 and len(sorted_group) == 1
                elif cur_task.comm_type in ['scatter', 'gather']:
                    root = int(cur_task.name.split('-')[1].split('_')[0].split(':')[-1])
                rets = OpCostModel.comm_cost(cur_task.comm_type,
                                             cur_task.volume,
                                             None,
                                             group=sorted_group,
                                             root=root,
                                             FlexFlow=self.flexflow)
                cur_task.set_cost(*rets)

                ready_t = cur_task.get_ready()
                for dev_id in cur_task.group:
                    ready_t = max(ready_t, devices[int(dev_id.split(':')[1])])
                end_t = cur_task.set_start_time(ready_t)
                if not cur_task.collective:
                    for dev_id in cur_task.group:
                        devices[int(dev_id.split(':')[1])] = end_t

            for ntsk in cur_task.next_tasks:
                ntsk.decrease_ndeps(cur_task.state)
                ntsk.ready[cur_task.state] = max(cur_task.end, ntsk.ready[cur_task.state])
                if ntsk.dep_free():
                    heapq.heappush(ready_queue, (ntsk.ready[cur_task.state], ntsk.id, ntsk))
        self.iter_speed = max(list(devices.values()))

    def print_stats(self):
        stats = {
            "Iteration Time": self.iter_speed,
            "Compile Time (s)": self.compile_time,
            "Run Time (s)": self.run_time,
            "Max Memory (MB)": 0
        }
        if self.flexflow:
            print()
            print('Iter Speed: {:.4f}ms/iter'.format(self.iter_speed))
            return stats

        print()
        print(
            'Iter Speed: {:.4f}ms/iter, Compile time: {:.4f}s, Run time: {:.4f}s'
            .format(self.iter_speed, self.compile_time, self.run_time))
        print('Compile time:')
        print('{}Task create: {:.4f}s'.format(' ' * 4, self.task_create_time))
        print('{}Dep analysis: {:.4f}s'.format(' ' * 4,
                                               self.dep_analysis_time))

        comm, other = 0, 0
        for i in range(Task.total_tasks()):
            if isinstance(Task.get(i), CommTask):
                comm += 1
            else:
                other += 1
        print('{}Comm Tasks: {}, Other Tasks: {}'.format(' ' * 4, comm, other))

        max_memory, dev_id = 0, -1
        for i, executor in self.executors.items():
            if executor.max_mem_alloc > max_memory:
                max_memory = executor.max_mem_alloc
                dev_id = i
        print('Max memory alloc: {:.3f}MB on device {}'.format(
            max_memory, dev_id))
        stats['Max Memory (MB)'] = max_memory
        print()

        return stats
