from proteus.strategy import DeviceTopo


class BaseAlgo(object):
    def __init__(self, dev_topo: DeviceTopo, scope):
        super().__init__()
        self.dev_topo = dev_topo
        self.scope = scope

    def optimize(self):
        raise NotImplementedError
