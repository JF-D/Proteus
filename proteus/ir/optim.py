class Optimizer(object):
    def __init__(self):
        super().__init__()


class SGD(Optimizer):
    def __init__(self,
                 model,
                 param,
                 lr=0.01,
                 momentum=0,
                 nesterov=False,
                 pconfig=None):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov

        attr = {'lr': lr, 'momentum': momentum, 'nesterov': nesterov}
        if momentum > 0:
            buf = model.Buffer(param.size(), dtype=param.dtype)
        else:
            buf = None
        self.op = model.SGD(param,
                            param.grad,
                            buf,
                            attr=attr,
                            pconfig=pconfig)


class Adam(Optimizer):
    def __init__(self, model, param, lr=0.01, betas=(0.9, 0.999), pconfig=None):
        super().__init__()
        self.lr = lr
        self.beta_1 = betas[0]
        self.beta_2 = betas[1]

        attr = {'lr': lr, 'beta1': self.beta_1, 'beta2': self.beta_2}
        exp_avg = model.Buffer(param.size(), dtype=param.dtype)
        exp_avg_sqr = model.Buffer(param.size(), dtype=param.dtype)
        self.op = model.Adam(param,
                             param.grad,
                             exp_avg,
                             exp_avg_sqr,
                             attr=attr,
                             pconfig=pconfig)
