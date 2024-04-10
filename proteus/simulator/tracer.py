import random
import json
import matplotlib.pyplot as plt
from collections import defaultdict


class SimTracer:

    def __init__(self, nprocess):
        self.trace_events = []
        self.memory_events = defaultdict(list)

        for pid in range(nprocess):
            # threads
            # 0: gpu, 1: cpu, 2: gpu2gpu, 3: cpu2gpu, 4: gpu2cpu
            thread_type = ['comp', 'cpu', 'gpu2gpu', 'grad', 'cpu2gpu', 'gpu2cpu']
            for tid in range(6):
                meta_event = {
                    'name': 'thread_name',
                    'ph': 'M',
                    'pid': pid,
                    'tid': tid,
                    'args': {
                        'name': thread_type[tid]
                    }
                }
                self.trace_events.append(meta_event)
                meta_event = {
                    'name': 'thread_sort_index',
                    'ph': 'M',
                    'pid': pid,
                    'tid': tid,
                    'args': {
                        'sort_index': pid * 3 + tid,
                    }
                }
                self.trace_events.append(meta_event)

    def add_trace_event(self, name, cat, pid, tid, times):
        for ph, ts in zip(['B', 'E'], times):
            event = {
                'name': name,
                'cat': cat,
                'ph': ph,
                'pid': pid,
                'tid': tid,
                'ts': ts * 1000
            }
            self.trace_events.append(event)

    def add_memory_event(self, dev_id, clock, memory):
        self.memory_events[dev_id].append((clock, memory))

    def export(self, trace_file):
        trace_data = {'traceEvents': self.trace_events, 'displayTimeUnit': 'ms'}
        with open(f'{trace_file}.json', 'w') as f:
            json.dump(trace_data, f)

        plt.figure(dpi=300)
        for dev_id, mevents in self.memory_events.items():
            x, y, total = [0], [0], 0
            for clock, memory in sorted(mevents, key=lambda x: x[0]):
                x.append(clock)
                y.append(total)
                total += memory
                x.append(clock)
                y.append(total)
            plt.plot(x,
                     y,
                     label='Dev-{}'.format(dev_id))
        plt.legend()
        plt.xlabel('Time (ms)')
        plt.ylabel('Memory (MB)')
        plt.savefig(f'{trace_file}.jpg')
