from proteus.ir.graph_builder import register_optimizer


class Optimzier(object):
    def __init__(self, params):
        super().__init__()
        param_groups = list(params)
        self.param_groups = [{'params': param_groups}]

    def step(self):
        raise NotImplementedError


class SGD(Optimzier):
    def __init__(self, params, lr=0.1, momentum=0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum

    def step(self):
        for group in self.param_groups:
            for param in group['params']:
                register_optimizer('SGD',
                                   param,
                                   lr=self.lr,
                                   momentum=self.momentum)


class Adam(Optimzier):
    def __init__(self, params, lr=0.1):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for group in self.param_groups:
            for param in group['params']:
                register_optimizer('Adam', param, lr=self.lr)
