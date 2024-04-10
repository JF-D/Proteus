from proteus.ir import Tensor, Parameter, Buffer
from . import functional as F
from .module import Module, register_module


def _pair(x):
    if not isinstance(x, (list, tuple)):
        return (x, x)
    else:
        return x


def _list_with_default(out_size, defaults):
    if isinstance(out_size, int):
        return out_size
    if len(defaults) <= len(out_size):
        raise ValueError(
            'Input dimension should be at least {}'.format(len(out_size) + 1))
    return [
        v if v is not None else d
        for v, d in zip(out_size, defaults[-len(out_size):])
    ]


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight = Parameter(Tensor(out_features, in_features,
                                       name='weight'))
        if bias:
            self.bias = Parameter(Tensor(out_features, name='bias'))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias, name=self.full_name)


class Conv2d(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 bias=True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        kernel_shape = (out_channels, in_channels, *self.kernel_size)
        self.weight = Parameter(Tensor(kernel_shape, name='weight'))
        if bias:
            self.bias = Parameter(Tensor(out_channels, name='bias'))
        else:
            self.bias = None

    def forward(self, input):
        return F.conv2d(input,
                        self.weight,
                        self.bias,
                        stride=self.stride,
                        padding=self.padding,
                        name=self.full_name)


class BatchNorm2d(Module):
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Parameter(Tensor(num_features, name='weight'))
            self.bias = Parameter(Tensor(num_features, name='bias'))
        else:
            self.weight = None
            self.bias = None
        # if self.track_running_stats:
        #     self.running_mean = Buffer(Tensor(num_features))
        #     self.running_var = Buffer(Tensor(num_features))
        # else:
        #     self.running_mean = None
        #     self.running_var = None

    def forward(self, input):
        return F.batch_norm(input,
                            weight=self.weight,
                            bias=self.bias,
                            training=(self.training
                                      or not self.track_running_stats),
                            momentum=self.momentum,
                            eps=self.eps,
                            name=self.full_name)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = Parameter(Tensor(normalized_shape, name='weight'))
            self.bias = Parameter(Tensor(normalized_shape, name='bias'))
        else:
            self.weight = None
            self.bias = None

    def forward(self, input):
        return F.layer_norm(input,
                            weight=self.weight,
                            bias=self.bias,
                            eps=self.eps,
                            name=self.full_name)


class ReLU(Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu(input, inplace=self.inplace, name=self.full_name)


class Activation(Module):
    def __init__(self, act, inplace=False):
        super().__init__()
        self.act_type = act
        self.inplace = inplace

    def forward(self, input):
        return F.activation(input,
                            self.act_type,
                            inplace=self.inplace,
                            name=self.full_name)


class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout(input,
                         p=self.p,
                         inplace=self.inplace,
                         name=self.full_name)


class MaxPool2d(Module):
    def __init__(self, kernel_size=1, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = self.kernel_size if stride is None else _pair(stride)
        self.padding = _pair(padding)

    def forward(self, input):
        return F.max_pool2d(input,
                            self.kernel_size,
                            stride=self.stride,
                            padding=self.padding,
                            name=self.full_name)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(AvgPool2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = self.kernel_size if stride is None else _pair(stride)
        self.padding = _pair(padding)

    def forward(self, input):
        return F.avg_pool2d(input,
                            self.kernel_size,
                            stride=self.stride,
                            padding=self.padding,
                            name=self.full_name)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, input):
        return F.adaptive_avg_pool2d(input,
                                     self.output_size,
                                     name=self.full_name)


class Softmax(Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.softmax(input, dim=self.dim, name=self.full_name)


class CrossEntropyLoss(Module):
    def __init__(self, loss_type=None):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, input, target):
        return F.cross_entropy(input, target, loss_type=self.loss_type, name=self.full_name)


class Elementwise(Module):
    def __init__(self, type):
        super().__init__()
        self.type = type

    def forward(self, *args):
        return F.elementwise(self.type, *args, name=self.full_name)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, attr=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.attr = attr

        self.weight = Parameter(
            Tensor(num_embeddings, embedding_dim, name='embedding'))

    def forward(self, input):
        return F.embedding(input, self.weight, name=self.full_name, attr=self.attr)


register_module(Linear.__name__, Linear)
register_module(Conv2d.__name__, Conv2d)
register_module(BatchNorm2d.__name__, BatchNorm2d)
register_module(LayerNorm.__name__, LayerNorm)
register_module(ReLU.__name__, ReLU)
register_module(Activation.__name__, Activation)
register_module(Dropout.__name__, Dropout)
register_module(MaxPool2d.__name__, MaxPool2d)
register_module(AvgPool2d.__name__, AvgPool2d)
register_module(AdaptiveAvgPool2d.__name__, AdaptiveAvgPool2d)
register_module(Softmax.__name__, Softmax)
register_module(CrossEntropyLoss.__name__, CrossEntropyLoss)
register_module(Elementwise.__name__, Elementwise)
register_module(Embedding.__name__, Embedding)
