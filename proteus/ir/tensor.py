import numbers
import operator
from functools import reduce

from proteus import DataType, size_of_datatype
from .graph_builder import register_op
from .node import Node


class Tensor(Node):
    """Tensor Node / Data Node.

    Tensor is n-dimensional array.
    """

    id = 0

    def __init__(self,
                 *args,
                 dtype=DataType.Float32,
                 id=None,
                 requires_grad=False,
                 name='t'):
        super().__init__()
        if isinstance(args[0], (tuple, list)):
            size = tuple(args[0])
        else:
            for i in args:
                assert isinstance(i, numbers.Integral)
            size = args
        self.shape = size
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.name = name
        self.grad = None
        self._is_parameter = False

        if id is None:
            self.id = Tensor.id
            Tensor.id += 1
        else:
            self.id = id

        self.producer = []
        self.consumer = []
        self.control = []

        # acount associated buffer that saved for backward usage
        # mainly used to account memory consumation
        self.buffer_account = 0

    def size(self, dim=None):
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    def dim(self):
        return len(self.shape)

    def set_requires_grad(self, requires_grad):
        self.requires_grad = requires_grad

    def set_grad(self, grad):
        if grad is not None:
            self.requires_grad = True
            self.grad = grad

    @property
    def is_leaf_grad(self):
        return False

    def add_producer(self, op, index):
        self.producer.append((op, index))

    def add_consumer(self, op, index):
        self.consumer.append((op, index))

    def partition(self, config):
        self.sub_tensors = {}
        for i in range(config.deg()):
            self.sub_tensors[i] = config.get_iters_interval(i)

    def set_associated_buffer(self, dtype, account):
        self.buffer_account = size_of_datatype(
            dtype) * account / size_of_datatype(self.dtype)

    @property
    def is_share_data(self):
        if getattr(self, 'share_from', None) is not None:
            return True
        return False

    @property
    def is_shared(self):
        if len(getattr(self, 'share_params', [])) > 0:
            return True
        return False

    # >>> Tensor operations begin
    def reshape(self, *shape):
        out = register_op('Reshape', [self], *shape)
        return out

    def view(self, *args):
        return self.reshape(*args)

    def view_as(self, other):
        return self.view(other.size())

    def flatten(self, start_dim=0, end_dim=-1):
        ndims = max(1, self.dim())
        assert -ndims <= start_dim < ndims and -ndims <= end_dim < ndims
        start_dim = start_dim % ndims
        end_dim = end_dim % ndims
        assert start_dim <= end_dim
        if self.dim() == 0:
            return self.reshape(())
        else:
            new_shape = (
                self.shape[:start_dim] +
                (reduce(operator.mul, self.shape[start_dim:end_dim + 1]), ) +
                self.shape[end_dim + 1:])
            return self.reshape(new_shape)

    def permute(self, *perm):
        out = register_op('Permute', [self], *perm)
        return out

    def split(self, size_or_sections, dim=0):
        outs = register_op('Split', [self], size_or_sections, dim=dim)
        return outs

    def slice(self, nelements):
        out = register_op('Slice', [self], nelements)
        return out

    # >>> Tensor operations end

    def __repr__(self):
        string = '%{}[{}]'.format(self.id, self.name)
        return string


def randn(*size, dtype=DataType.Float32):
    """Make a rand tensor."""
    if isinstance(size[0], (tuple, list)):
        size = size[0]
    out = Tensor(size, dtype=dtype)
    return out


class Input(Tensor):

    def __init__(self, data):
        super().__init__(data.size(),
                         dtype=data.dtype,
                         id=data.id,
                         requires_grad=data.requires_grad,
                         name=data.name)


class Output(Tensor):

    def __init__(self, data):
        super().__init__(data.size(),
                         dtype=data.dtype,
                         id=data.id,
                         requires_grad=data.requires_grad,
                         name=data.name)


class Parameter(Tensor):

    def __init__(self, data, requires_grad=True):
        super().__init__(data.size(),
                         dtype=data.dtype,
                         id=data.id,
                         requires_grad=data.requires_grad,
                         name=data.name)
        self.set_requires_grad(requires_grad)
        self._is_parameter = True

        self.share_params = []
        self.share_from = None

        self.optimizer = None

    def set_optimizer(self, op):
        self.optimizer = op


class Gradient(Tensor):

    def __init__(self, data, original_tensor):
        super().__init__(data.size(),
                         dtype=data.dtype,
                         id=data.id,
                         requires_grad=data.requires_grad,
                         name=data.name)
        self._is_grad = True
        self._is_leaf_grad = isinstance(original_tensor, Parameter)

        self.org_id = original_tensor.id
        original_tensor.set_grad(self)

        self.share_params = []
        self.share_from = None

    @property
    def is_leaf_grad(self):
        return self._is_leaf_grad


class Buffer(Tensor):

    def __init__(self, data):
        super().__init__(data.size(),
                         dtype=data.dtype,
                         id=data.id,
                         requires_grad=data.requires_grad,
                         name=data.name)
