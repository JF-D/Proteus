from proteus import DevType, MapType
from proteus.ir import ProteusModel, optim
from proteus.strategy.device import Device, DeviceTopo
from proteus.strategy import TensorConfig, OpConfig


def build_dev_N4():
    g0 = Device(0, DevType.GPU, 16000, 15700, 900)
    g1 = Device(1, DevType.GPU, 16000, 15700, 900)
    g2 = Device(2, DevType.GPU, 16000, 15700, 900)
    g3 = Device(3, DevType.GPU, 16000, 15700, 900)
    bandwidth = [[748, 48, 48, 12], [48, 748, 96, 12], [48, 96, 748, 12],
                 [12, 12, 12, 748]]
    dev_topo = DeviceTopo([g0, g1, g2, g3], bandwidth)
    return dev_topo


def build_dev_N2():
    g0 = Device(0, DevType.GPU, 16000, 15700, 900)
    g1 = Device(1, DevType.GPU, 16000, 15700, 900)
    bandwidth = [[748, 48], [48, 748]]
    dev_topo = DeviceTopo([g0, g1], bandwidth)
    return dev_topo


def main():
    # build model
    dev_topo = build_dev_N2()
    model = ProteusModel(dev_topo, train=True)
    x = model.Placeholder((32, 500))
    # label = model.Placeholder((32, ), is_label=True)
    y = model.Linear(x, 1000, use_bias=False)
    # model.CrossEntropyLoss(y, label)

    # build optimizer
    # optim.SGD(model, lr=0.01)

    # visiualize
    model.to_graphviz()

    # parallel
    # dev_topo = build_dev_N4()
    # config = model.parallel_config(dev_topo, max_parts=8, stride=2)
    # mapping = [(MapType.SHARD, 0), (MapType.SHARD, 1), (MapType.SHARD, 2), (MapType.SHARD, 3)]
    # model.partition_and_map(x, ((4, 1), mapping))
    # model.partition_and_map(y, ((4, 1), mapping))
    # model.symmetric_forward_backward()

    config = model.parallel_config(dev_topo, max_parts=4, stride=2)
    mapping = [(MapType.SHARD, 0), (MapType.SHARD, 1)]
    model.partition_and_map(x, ((2, 1), mapping))
    model.partition_and_map(y, ((2, 1), mapping))
    model.symmetric_forward_backward()

    model.simulate()
    model.export_config('config.txt')
    model.task_manager.to_graphviz()

    schedule = model.task_manager.get_schedule()
    print(schedule)
    ret = model.task_manager.evaluate_strategy()
    print('time cost: {:.2f}ms, {}'.format(
        ret[0], 'within memory' if ret[1] else 'exceed memory'))


if __name__ == '__main__':
    main()
