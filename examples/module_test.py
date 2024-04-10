from proteus.nn.cube import DeviceCube
import proteus.nn as nn
from proteus import DevType
from proteus.ir import ProteusModel, graph
from proteus.strategy.device import Device, DeviceTopo


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 400, bias=False)
        self.fc2 = nn.Linear(400, 100, bias=False)

    def forward(self, graph, x):
        x = self.fc1(graph, x)
        x = self.fc2(graph, x)
        return x


class MixMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = MLP()
        self.mlp2 = MLP()

    def forward(self, graph, x):
        x = self.mlp1(graph, x)
        x = self.mlp2(graph, x)
        return x


def build_dev_topo():
    g0 = Device(0, DevType.GPU, 16000, 15700, 900)
    g1 = Device(1, DevType.GPU, 16000, 15700, 900)
    g2 = Device(2, DevType.GPU, 16000, 15700, 900)
    g3 = Device(3, DevType.GPU, 16000, 15700, 900)
    bandwidth = [[748, 48, 48, 12], [48, 748, 96, 12], [48, 96, 748, 12],
                 [12, 12, 12, 748]]
    dev_topo = DeviceTopo([g0, g1, g2, g3], bandwidth)
    return dev_topo


if __name__ == '__main__':
    # algorithm
    model = nn.Sequential(MixMLP(), MixMLP())
    criterion = nn.CrossEntropyLoss()
    optimizer = nn.optim.SGD(lr=0.1)

    cube = [[[0, 1], [2, 3]]]
    # cube = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    dev_cube = DeviceCube(list(range(4)), cube)
    model.split(dev_cube, 1)
    MLP.split('fc1', 1, 'MP')
    MLP.split('fc2', 2, 'MP')

    dev_topo = build_dev_topo()
    graph = ProteusModel(dev_topo, train=True)
    x = graph.Placeholder((32, 100))
    label = graph.Placeholder((32, ))

    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential):
            print(name, module.mesh)
            if isinstance(module, nn.BuiltinModule):
                print(module.splited)
    # real graph building process
    y = model(graph, x)
    criterion(graph, y, label)
    optimizer.step(graph)

    graph.to_graphviz()

    config = graph.parallel_config(max_parts=4, stride=2)
    graph.symmetric_forward_backward()

    graph.simulate()
    graph.export_config('config.txt')
    graph.task_manager.to_graphviz()

    schedule = graph.task_manager.get_schedule()
    print(schedule)
    ret = graph.task_manager.evaluate_strategy()
    print('time cost: {:.2f}ms, {}'.format(
        ret[0], 'within memory' if ret[1] else 'exceed memory'))
