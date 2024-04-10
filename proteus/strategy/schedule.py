import math
import weakref
from queue import PriorityQueue
from collections import defaultdict
from proteus.simulator.task import MemTask, CompTask, CommTask


class Schedule(object):
    def __init__(self, task_manager):
        super().__init__()
        self.tmg_ref = weakref.ref(task_manager)
        self.exec_plan = defaultdict(list)
        self.order = []
        self.measured_time = False

    def tasks(self, id):
        return self.tmg_ref().tasks[id]

    def add_task(self, task_id):
        self.order.append(task_id)
        task = self.tasks(task_id)
        if isinstance(task, CommTask):
            self.exec_plan[task.src].append(task_id)
            self.exec_plan[task.dst].append(task_id)
        else:
            if task.device >= 0:
                self.exec_plan[task.device].append(task_id)

    def measure_time_cost(self):
        # initial ready time
        for tid in self.order:
            self.tasks(tid).set_ready_time(0.0)
        # simulator
        dev_clock = dict.fromkeys(self.exec_plan.keys(), 0.0)
        for tid in self.order:
            task = self.tasks(tid)
            if isinstance(task, MemTask):
                for ntid in task.next_tasks:
                    if self.tasks(ntid).ready < task.ready:
                        self.tasks(ntid).set_ready_time(task.ready)
            elif isinstance(task, CompTask):
                start = max(dev_clock[task.device], task.ready)
                task.set_start_time(start)
                dev_clock[task.device] = task.end
                for ntid in task.next_tasks:
                    if self.tasks(ntid).ready < task.end:
                        self.tasks(ntid).set_ready_time(task.end)
            else:
                if (task.src, task.dst) not in dev_clock:
                    dev_clock[(task.src, task.dst)] = 0.0
                start = max(dev_clock[(task.src, task.dst)], task.ready)
                task.set_start_time(start)
                dev_clock[(task.src, task.dst)] = task.end
                for ntid in task.next_tasks:
                    if self.tasks(ntid).ready < task.end:
                        self.tasks(ntid).set_ready_time(task.end)
        total_cost = max(dev_clock.values())
        self.measured_time = True
        return total_cost

    def measure_memory_cost(self):
        dev_mem_cost = dict.fromkeys(self.exec_plan.keys(), 0.0)
        max_mem_cost = dict.fromkeys(self.exec_plan.keys(), 0.0)
        # set alive time
        inter_mem_tasks = dict.fromkeys(self.exec_plan.keys(), list())
        if not self.measured_time:
            self.measure_time_cost()
        for tid in self.order:
            task = self.tasks(tid)
            if not isinstance(task, MemTask):
                continue
            if task.permanent:
                task.set_start_time(0.0)
                task.set_end_time(math.inf)
                dev_mem_cost[task.device] += task.memory
            else:
                start = min([self.tasks(t).start for t in task.prev_tasks])
                end = max([self.tasks(t).end
                           for t in task.next_tasks] + [start + 0.0001])
                task.set_start_time(start)
                task.set_end_time(end)
                if task.device >= 0:
                    inter_mem_tasks[task.device].append(task)
                else:
                    assert len(task.prev_tasks) > 0
                    ptask = self.tasks(task.prev_tasks[0])
                    if isinstance(ptask, CommTask):
                        device = ptask.dst
                    else:
                        device = ptask.device
                    inter_mem_tasks[device].append(task)
        # max memory analysis
        for dev, dev_tasks in inter_mem_tasks.items():
            mtasks = sorted(dev_tasks, key=lambda task: task.start)
            task_queue = PriorityQueue()
            for task in mtasks:
                while not task_queue.empty():
                    if task_queue.queue[0][0] <= task.start:
                        _, front_task = task_queue.get()
                        dev_mem_cost[dev] -= front_task.memory
                    else:
                        break
                task_queue.put((task.end, task))
                dev_mem_cost[dev] += task.memory
                if dev_mem_cost[dev] > max_mem_cost[dev]:
                    max_mem_cost[dev] = dev_mem_cost[dev]
        return max_mem_cost

    def __repr__(self) -> str:
        string = ''
        for dev in self.exec_plan.keys():
            task_names = []
            for tid in self.exec_plan[dev]:
                task = self.tasks(tid)
                if isinstance(task, CommTask):
                    task_names.append(task.name + ':' + str(task.id))
                else:
                    task_names.append(task.name)
            plan = ', '.join(task_names)
            string += '{}: {}\n'.format(dev, plan)
        return string
