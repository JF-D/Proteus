import numpy as np
from .utils import flat_to_coordinate, get_strides


class Interval(object):
    def __init__(self, lb, ub):
        super().__init__()
        self.lb = lb
        self.ub = ub
        self.size = tuple([u - l for u, l in zip(ub, lb)])

    def get_subinterval(self, indices):
        lb, ub = [], []
        for i in indices:
            lb.append(self.lb[i])
            ub.append(self.ub[i])
        return Interval(lb, ub)

    def nelements(self):
        extent = 1
        for l, u in zip(self.lb, self.ub):
            extent *= u - l
        return extent

    def __len__(self):
        return len(self.lb)

    def __eq__(self, other):
        return self.lb == other.lb and self.ub == other.ub

    def intersection(self, other):
        assert len(self) == len(other)
        intersect = []
        for i in range(len(self)):
            l = min(self.ub[i], other.ub[i]) - max(self.lb[i], other.lb[i])
            if l <= 0:
                return 0
            intersect.append(l)
        return np.prod(intersect)

    def __repr__(self) -> str:
        string = '['
        for l, u in zip(self.lb, self.ub):
            string += '{}:{}, '.format(l, u)
        string += ']'
        return string


class InputSpace(object):
    def __init__(self, infer_type, coeffs_or_indices=None):
        super().__init__()
        self.infer_type = infer_type
        if self.infer_type == 'Identity':
            self.indices = coeffs_or_indices
        elif self.infer_type == 'Infer':
            self.coeffs = coeffs_or_indices
        else:
            assert False, 'Unknown infer_type: {}'.format(infer_type)

    def input_interval(self, idx, iters_interval: Interval):
        if self.infer_type == 'Identity':
            return iters_interval.get_subinterval(self.indices[idx])

    @staticmethod
    def get_identity(iter_space):
        indices = iter_space.in_iters
        return InputSpace('Identity', indices)


def get_coordinate_interval(coordinate, parts, bounds):
    lbs, ubs = [], []
    for i, (p, k) in enumerate(zip(parts, coordinate)):
        extent = bounds[i] // p
        lb = extent * k
        if bounds[i] % p > k:
            lb += k
            extent += 1
        else:
            lb += bounds[i] % p
        lbs.append(lb)
        ubs.append(lb + extent)
    return Interval(lbs, ubs)


def get_iters_interval(flat_idx, parts, bounds, strides=None):
    if strides is None:
        strides = get_strides(parts)
    coordinate = flat_to_coordinate(flat_idx, strides)
    return get_coordinate_interval(coordinate, parts, bounds)
