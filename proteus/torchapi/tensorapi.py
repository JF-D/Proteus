from proteus import DataType
from proteus.ir.graph_builder import register_op, register_placeholder


def zeros(*size, dtype=DataType.Float32, requires_grad=False):
    if isinstance(size[0], (tuple, list)):
        size = size[0]
    out = register_placeholder('Placeholder',
                               size,
                               dtype=dtype,
                               requires_grad=requires_grad)
    return out


def randn(*size, dtype=DataType.Float32, requires_grad=False):
    if isinstance(size[0], (tuple, list)):
        size = size[0]
    out = register_placeholder('Placeholder',
                               size,
                               dtype=dtype,
                               requires_grad=requires_grad)
    return out


def matmul(input, other):
    out = register_op('Matmul', [input, other])
    return out


def reshape(input, *shape):
    return input.reshape(*shape)


def view(input, *args):
    return input.view(*args)


def flatten(input, start_dim=0, end_dim=-1):
    return input.flatten(start_dim=start_dim, end_dim=end_dim)


def permute(input, *perm):
    return input.permute(*perm)


def split(input, size_or_sections, dim=0):
    return input.split(size_or_sections, dim=dim)


def cat(inputs, dim=0):
    out = register_op('Concat', [inputs], dim=dim)
    return out


def concat(inputs, dim=0):
    return cat(inputs, dim=dim)
