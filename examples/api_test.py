import argparse
from proteus.nn.cube import DeviceCube
import proteus.nn as nn
from proteus import DevType
from proteus.ir import ProteusModel, graph
from proteus.strategy.device import Device, DeviceTopo

parser = argparse.ArgumentParser()
parser.add_argument('-ps', type=str, default='dp')
args = parser.parse_args()


class FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 100, bias=False)

    def forward(self, graph, x):
        x = self.fc(graph, x)
        return x


def build_dev_topo_n2():
    g0 = Device(0, DevType.GPU, 16000, 15700, 900)
    g1 = Device(1, DevType.GPU, 16000, 15700, 900)
    bandwidth = [[748, 48], [48, 748]]
    dev_topo = DeviceTopo([g0, g1], bandwidth)
    return dev_topo


def build_dev_topo_n4():
    g0 = Device(0, DevType.GPU, 16000, 15700, 900)
    g1 = Device(1, DevType.GPU, 16000, 15700, 900)
    g2 = Device(2, DevType.GPU, 16000, 15700, 900)
    g3 = Device(3, DevType.GPU, 16000, 15700, 900)
    bandwidth = [[748, 48, 48, 12], [48, 748, 96, 12], [48, 96, 748, 12],
                 [12, 12, 12, 748]]
    dev_topo = DeviceTopo([g0, g1, g2, g3], bandwidth)
    return dev_topo


if __name__ == '__main__':
    model = nn.Sequential(FC(), FC())

    if args.ps == 'dp':
        cube = [[[0, 1]]]
        dev_cube = DeviceCube(list(range(2)), cube)
        dev_topo = build_dev_topo_n2()

        model.split(dev_cube, 1)
        FC.split('fc', 0, 'MP')
    elif args.ps == 'mp':
        cube = [[[0, 1]]]
        dev_cube = DeviceCube(list(range(2)), cube)
        dev_topo = build_dev_topo_n2()

        model.split(dev_cube, 1)
        getattr(model, '0').fc.split(1, 'MP')
        getattr(model, '1').fc.split(2, 'MP')
    elif args.ps == 'pp':
        cube = [[[0], [1]], [[2], [3]]]
        dev_cube = DeviceCube(list(range(4)), cube)
        dev_topo = build_dev_topo_n4()

        model.split(dev_cube, 2)
        FC.split('fc', 0, 'MP')
    elif args.ps == 'zero':
        model = nn.Sequential(FC())

        cube = [[[0, 1], [2, 3]]]
        dev_cube = DeviceCube(list(range(4)), cube)
        dev_topo = build_dev_topo_n4()

        model.split(dev_cube, 1)
        FC.split('fc', 1, 'MP')
        FC.split('fc', 0, 'DP', 'weight')

    graph = ProteusModel(dev_topo, train=False)
    x = graph.Placeholder((32, 100))

    y = model(graph, x)

    graph.to_graphviz()

    config = graph.parallel_config(max_parts=4, stride=2)
    graph.symmetric_forward_backward()

    graph.simulate()
    graph.export_config('config.txt')
    graph.task_manager.to_graphviz()

    schedule = graph.task_manager.get_schedule()
    ret = graph.task_manager.evaluate_strategy()
    print('time cost: {:.2f}ms, max memory: {:.2f}MB, {}'.format(
        ret[0], ret[2], 'within memory' if ret[1] else 'exceed memory'))
