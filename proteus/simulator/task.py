from proteus import TaskType


class Task(object):
    id = 0

    def __init__(self, type, name):
        super().__init__()
        self.type = type
        self.name = name
        self.next_tasks = []
        self.prev_tasks = []

        self.id = Task.id
        Task.id += 1

    def add_next_task(self, task):
        self.next_tasks.append(task.id)

    def add_prev_tasks(self, task):
        self.prev_tasks.append(task.id)

    def set_ready_time(self, time):
        self.ready = time

    def __lt__(self, other):
        # dummy cmp operator, used to avoid error in PriorityQueue.
        return self.id < other.id


class MemTask(Task):
    def __init__(self, name, memory):
        super().__init__(TaskType.TASK_MEM, name)
        self.cost = 0
        self.memory = memory
        self.permanent = False

    def set_permanent(self):
        self.permanent = True

    def set_start_time(self, start):
        self.start = start

    def set_end_time(self, end):
        self.end = end


class CompTask(Task):
    def __init__(self, name, cost):
        super().__init__(TaskType.TASK_COMP, name)
        self.cost = cost

    def set_start_time(self, time):
        self.start = time
        self.end = self.start + self.cost


class CommTask(Task):
    def __init__(self, name, cost):
        super().__init__(TaskType.TASK_COMM, name)
        self.cost = cost
        self.name = '{}_{}'.format(self.name, self.id)

    def set_start_time(self, time):
        self.start = time
        self.end = self.start + self.cost
