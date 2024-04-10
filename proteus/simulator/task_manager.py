import json
import math
from collections import defaultdict
from queue import PriorityQueue
from graphviz import Digraph
from proteus import size_of_datatype, enum_to_str, OpType
from proteus.ir.tensor import Parameter, Gradient, Input, Buffer
from proteus.strategy import Schedule
from .task import MemTask, CompTask, CommTask
from .cost_model import OpCostModel


class TaskManager(object):
    def __init__(self):
        super().__init__()
        self.tasks = {}
        self.cost_model = OpCostModel()

    def set_dev_topo(self, dev_topo):
        self.dev_topo = dev_topo

    def new_mem_task(self, data, index, dev_id):
        name = '{}_{}/{}:{}'.format(data.name, data.id, index, dev_id)
        mem = data.sub_tensors[index].nelements() * size_of_datatype(
            data.dtype) / 1024 / 1024
        task = MemTask(name, mem)
        if isinstance(data, (Parameter, Gradient, Input, Buffer)):
            task.set_permanent()
        task.device = dev_id
        self.tasks[task.id] = task
        return task

    def new_comp_task(self, op, index, dev_id):
        name = '{}_{}/{}:{}'.format(enum_to_str(OpType, op.type), op.id, index,
                                    dev_id)
        dev = self.dev_topo.dev(dev_id)
        cost = op.measure_cost(dev, 'profile') / len(op.sub_ins)
        task = CompTask(name, cost)
        task.device = dev_id
        self.tasks[task.id] = task
        return task

    def new_comm_task(self, nelements, dtype, src, dst, name='xfer'):
        name = name
        mbytes = nelements * size_of_datatype(dtype) / 1024 / 1024
        cost = OpCostModel.comm_cost('p2p', mbytes, self.dev_topo.bw(src, dst))
        task = CommTask(name, cost)
        task.src = src
        task.dst = dst
        self.tasks[task.id] = task
        return task

    def get_schedule(self):
        # set prev tasks
        for _, task in self.tasks.items():
            for ntid in task.next_tasks:
                self.tasks[ntid].add_prev_tasks(task)

        # set depth level
        topo_order = []
        visited = dict.fromkeys(self.tasks.keys(), False)

        def DFS(task_id):
            if visited[task_id]:
                return
            visited[task_id] = True
            for ntask_id in self.tasks[task_id].next_tasks:
                DFS(ntask_id)
            topo_order.append(task_id)

        for task_id in self.tasks.keys():
            DFS(task_id)
        depth = defaultdict(lambda: 0)
        for task_id in topo_order:
            depth[task_id] += self.tasks[task_id].cost
            for ptask_id in self.tasks[task_id].prev_tasks:
                depth[ptask_id] = max(depth[ptask_id], depth[task_id])
        topo_order.reverse()

        # get in-degree
        in_deg = defaultdict(int)
        for k, task in self.tasks.items():
            if k not in in_deg.keys():
                in_deg[k] = 0
            for dst in task.next_tasks:
                in_deg[dst] += 1
        initial_tasks = []
        for k, v in in_deg.items():
            if v == 0:
                initial_tasks.append(k)

        # depth-first schedule
        task_queue = PriorityQueue()
        self.schedule = Schedule(self)
        for task_id in initial_tasks:
            task_queue.put((-depth[task_id], task_id))
        while not task_queue.empty():
            _, task_id = task_queue.get()
            for ntask_id in self.tasks[task_id].next_tasks:
                in_deg[ntask_id] -= 1
                if in_deg[ntask_id] == 0:
                    task_queue.put((-depth[ntask_id], ntask_id))
            self.schedule.add_task(task_id)
        return self.schedule

    def evaluate_strategy(self, schedule=None):
        if schedule is None:
            schedule = self.get_schedule()
        time_cost = schedule.measure_time_cost()
        mem_cost = schedule.measure_memory_cost()
        for dev, m in mem_cost.items():
            if m > 0.85 * self.dev_topo.dev(dev).memory:
                return time_cost, False, max(mem_cost.values())
        return time_cost, True, max(mem_cost.values())

    def to_trace(self, trace_file='timeline.json'):
        trace_events = list()
        for _, task in self.tasks.items():
            if isinstance(task, CompTask):
                for ph, ts in zip(['B', 'E'], [task.start, task.end]):
                    event = {
                        'name': task.name,
                        'cat': task.name.split('_')[0],
                        'ph': ph,
                        'pid': task.device,
                        'tid': 0,
                        'ts': ts * 1000
                    }
                    trace_events.append(event)
            elif isinstance(task, CommTask):
                for ph, ts in zip(['B', 'E'], [task.start, task.end]):
                    event_src = {
                        'name': '{}/{}:{}'.format(task.name, task.src,
                                                  task.dst),
                        'cat': task.name,
                        'ph': ph,
                        'pid': task.src,
                        'tid': 1,
                        'ts': ts * 1000
                    }
                    event_dst = {
                        'name': '{}_{}:{}'.format(task.name, task.src,
                                                  task.dst),
                        'cat': task.name,
                        'ph': ph,
                        'pid': task.dst,
                        'tid': 1,
                        'ts': ts * 1000
                    }
                    trace_events.append(event_src)
                    trace_events.append(event_dst)
            elif isinstance(task, MemTask):
                event = {
                    'name': task.name,
                    'ph': 'I',
                    'pid': task.device,
                    'tid': 0,
                    'ts': task.ready * 1000
                }
                trace_events.append(event)
            for ntid in task.next_tasks:
                ntsk = self.tasks[ntid]
                event_st = {
                    'name':
                    'connect',
                    'ph':
                    's',
                    'id':
                    '{}_{}'.format(task.id, ntsk.id),
                    'pid':
                    task.dst if isinstance(task, CommTask) else task.device,
                    'tid':
                    1 if isinstance(task, CommTask) else 0,
                    'ts':
                    (task.ready if isinstance(task, MemTask) else task.end) *
                    1000,
                }
                event_ed = {
                    'name':
                    'connect',
                    'ph':
                    'f',
                    'id':
                    '{}_{}'.format(task.id, ntsk.id),
                    'bp':
                    'e',
                    'pid':
                    ntsk.src if isinstance(ntsk, CommTask) else ntsk.device,
                    'tid':
                    1 if isinstance(ntsk, CommTask) else 0,
                    'ts':
                    (ntsk.ready if isinstance(ntsk, MemTask) else ntsk.start) *
                    1000,
                }
                trace_events.append(event_st)
                trace_events.append(event_ed)
        trace_data = {'traceEvents': trace_events, 'displayTimeUnit': 'ms'}
        with open(trace_file, 'w') as f:
            json.dump(trace_data, f)

    def to_graphviz(self, file='log/sim_graph'):
        graph = Digraph()
        for k, v in self.tasks.items():
            if isinstance(v, MemTask):
                graph.node(str(k), v.name, shape='box')
            elif isinstance(v, CompTask):
                graph.node(str(k), v.name)
            elif isinstance(v, CommTask):
                graph.node(str(k), v.name + ':' + str(k), shape='diamond')
        for k, v in self.tasks.items():
            for dst in v.next_tasks:
                graph.edge(str(v.id), str(dst))
        with open(f'{file}.txt', 'w') as f:
            f.write(graph.source)
