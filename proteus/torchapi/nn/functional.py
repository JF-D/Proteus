from proteus.ir.graph_builder import register_op


def linear(input, weight, bias=None, name=''):
    ins = [input, weight, bias] if bias else [input, weight]
    out = register_op('Linear', ins, name=name)
    return out


def conv2d(input, weight, bias=None, stride=1, padding=0, name=''):
    ins = [input, weight, bias] if bias else [input, weight]
    out = register_op('Conv2d', ins, stride=stride, padding=padding, name=name)
    return out


def batch_norm(input,
               weight=None,
               bias=None,
               training=False,
               momentum=0.1,
               eps=1e-5,
               name=''):
    ins = [input, weight, bias] if weight else [input]
    out = register_op('BatchNorm2d', ins, name=name)
    return out


def layer_norm(input, weight=None, bias=None, eps=1e-5, name=''):
    ins = [input, weight, bias] if weight else [input]
    out = register_op('LayerNorm', ins, name=name)
    return out


def relu(input, inplace=False, name=''):
    out = register_op('ReLU', [input], name=name)
    return out


def activation(input, act_type, inplace=False, name=''):
    out = register_op('Activation', [input], act_type=act_type, name=name)
    return out


def dropout(input, p=0.5, inplace=False, name=''):
    out = register_op('Dropout', [input], p=p, name=name)
    return out


def max_pool2d(input, kernel_size, stride=None, padding=0, name=''):
    out = register_op('Pool2d', [input],
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      mode='max',
                      name=name)
    return out


def avg_pool2d(input, kernel_size, stride=None, padding=0, name=''):
    out = register_op('Pool2d', [input],
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      mode='avg',
                      name=name)
    return out


def adaptive_avg_pool2d(input, output_size, name=''):
    out = register_op('AdaptivePool2d', [input],
                      output_size,
                      mode='avg',
                      name=name)
    return out


def softmax(input, dim=0, name=''):
    out = register_op('Softmax', [input], dim=dim, name=name)
    return out


def cross_entropy(input, target, loss_type=None, name=''):
    out = register_op('CrossEntropyLoss', [input, target], loss_type=loss_type, name=name)
    return out


def elementwise(type, *args, name=''):
    out = register_op('Elementwise', args, type, name=name)
    return out


def embedding(input, weight, name='', attr=None):
    out = register_op('Embedding', [input, weight], name=name, attr=attr)
    return out
