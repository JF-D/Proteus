import random
import numpy as np
from proteus import IterType, MapType, int_to_enum
from proteus.utils import IntSet, get_strides, get_iters_interval


class Config(object):

    def __init__(self, ndevs, ranges, bounds, id):
        super().__init__()
        self.ndevs = ndevs
        self.ranges = tuple(ranges)
        self.bounds = bounds
        self.id = id

        self.manual = False

        self.parts = [1] * len(self.ranges)
        self.strides = get_strides(self.parts)
        self.mapping = []
        self.mesh = np.array([])

    def deg(self):
        return np.prod(self.parts)

    def get_part(self, idx):
        return self.parts[idx]

    def set_config(self):
        raise NotImplementedError

    def get_config(self):
        return tuple(self.parts), self.mapping, self.mesh

    def manual_pconfig(self):
        raise NotImplementedError

    def get_iters_interval(self, flat_idx):
        return get_iters_interval(flat_idx, self.parts, self.bounds,
                                  self.strides)

    def export(self, f):
        f.write('{:30}, '.format(str(self.parts)))
        f.write(str(self.mapping) + f' ({self.manual})' + '\n')

    @property
    def dev_type(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        string = '{:30}, {}, ({})'.format(str(self.parts), self.mapping,
                                          self.manual)
        return string


class TensorConfig(Config):

    def __init__(self,
                 data,
                 ndevs,
                 max_parts=None,
                 stride=1,
                 allow_drop=False):
        _ranges = []
        max_parts = max_parts if max_parts else ndevs
        for s in data.size():
            _ranges.append(
                IntSet(1, min(s, max_parts), stride_or_candidates=stride))
        super().__init__(ndevs, _ranges, tuple(data.size()), data.id)
        self.init_tensor_config()

        self.recompute = False

    def init_tensor_config(self):
        self.parts = [1] * len(self.ranges)
        self.strides = get_strides(self.parts)
        self.mapping = [(MapType.SHARD, 0)]

    def set_config(self, config):
        # config: (parts, mapping, mesh) tuple
        assert len(config[0]) == len(self.bounds)
        assert np.prod(config[0]) == np.prod(config[2].shape[:-1])
        self.parts = config[0]
        self.mapping = config[1]
        self.mesh = config[2]
        self.strides = get_strides(self.parts)

    def is_recompute(self):
        return self.recompute

    def manual_pconfig(self, pconfig):
        if 'recompute' in pconfig:
            self.recompute = True
            if len(pconfig) == 1:
                return
        cfg = pconfig['partition']
        self.parts = [1] * len(self.ranges)
        for k, v in cfg.items():
            self.parts[k] = v
        self.parts = tuple(self.parts)
        self.strides = get_strides(self.parts)
        for k, cfg in pconfig.items():
            if k == 'partition':
                continue
            if k == 'map':
                self.mapping = cfg.copy()
            elif k == 'recompute':
                self.recompute = True
            elif k == 'mesh':
                mesh_size = list(self.parts) + [-1]
                self.mesh = cfg.reshape(mesh_size)
        self.manual = True

    def random(self):
        if self.manual:
            return self.parts, self.mapping
        # partition
        self.parts = tuple([r.random() for r in self.ranges])
        while np.prod(self.parts) > self.ndevs:
            self.parts = tuple([r.random() for r in self.ranges])
        self.parts = tuple([1 for _ in self.ranges])
        self.strides = get_strides(self.parts)
        # device mapping
        self.mapping = []
        for _ in range(np.prod(self.parts)):
            n = random.randint(1, min(2, self.ndevs))
            state = int_to_enum(MapType, n)
            if state == MapType.DROP:
                mapping = (state, )
            elif state == MapType.SHARD:
                mapping = (state, random.randint(0, self.ndevs - 1))
            elif state == MapType.REPLICATE:
                mapping = [state]
                map_num = random.randint(2, self.ndevs)
                map_devs = random.sample(range(self.ndevs), map_num)
                mapping = tuple(mapping + map_devs)
            self.mapping.append(mapping)
        return self.parts, self.mapping

    @property
    def dev_type(self):
        if len(self.mapping) == 0 or self.mapping[0][0] == MapType.DROP:
            return None
        dev_type = self.mapping[0][1].split(':')[0]
        for dev in self.mapping:
            assert dev_type == dev[1].split(':')[0]
        return dev_type

    def export(self, f):
        f.write('{:30}, '.format(str(self.parts)))
        f.write(str(self.mapping) + f' {self.recompute}, ({self.manual})' + '\n')

    def __repr__(self) -> str:
        string = '{:30}, {}, {}, ({})'.format(str(self.parts), self.mapping,
                                              self.recompute, self.manual)
        return string

class OpConfig(Config):

    def __init__(self, op, ndevs, max_parts=None, stride=1):
        iter_space = op.get_iter_space()
        _ranges = []
        max_parts = max_parts if max_parts else ndevs
        for i, s in enumerate(iter_space.bounds):
            if iter_space.iters[i] == IterType.OPAQUE:
                _ranges.append(IntSet(1, 1, stride_or_candidates=1))
            elif iter_space.iters[i] == IterType.CONTINUOUS:
                # # this is for reshape op
                # in_dims = [
                #     op.ins[0][k] for k in range(len(op.ins[0]))
                #     if iter_space.in_iters[0][k] == i
                # ]
                # out_dims = [
                #     op.outs[0][k] for k in range(len(op.outs[0]))
                #     if iter_space.out_iters[0][k] == i
                # ]
                # p_sets = [set(), set()]
                # for p, dims in enumerate([in_dims, out_dims]):
                #     acc = 1
                #     for d in dims:
                #         for k in range(1, d + 1):
                #             if acc * k > max_parts:
                #                 break
                #             if acc * k == 1 or acc * k % stride == 0:
                #                 p_sets[p].add(acc * k)
                #         acc *= d
                #         if acc > max_parts:
                #             break
                # p_set = p_sets[0].intersection(p_sets[1])
                # p_set = sorted(list(p_set))
                # _ranges.append(
                #     IntSet(p_set[0], p_set[-1], stride_or_candidates=p_set))

                _ranges.append(
                    IntSet(1, min(s, max_parts), stride_or_candidates=stride))
            else:
                _ranges.append(
                    IntSet(1, min(s, max_parts), stride_or_candidates=stride))
        super().__init__(ndevs, _ranges, tuple(iter_space.bounds), op.id)
        self.init_op_config()

    def init_op_config(self):
        self.parts = [1] * len(self.ranges)
        self.strides = get_strides(self.parts)
        self.mapping = [(MapType.SHARD, 0)]

    def set_config(self, config):
        # config: (parts, mapping, mesh) tuple
        assert len(config[0]) == len(self.bounds)
        assert np.prod(config[0]) == np.prod(config[2].shape[:-1])
        self.parts = config[0]
        self.mapping = config[1]
        self.mesh = config[2]
        self.strides = get_strides(self.parts)

    def manual_pconfig(self, pconfig):
        self.parts = [1] * len(self.ranges)
        if len(pconfig['partition']) == 0:
            self.manual = False
            self.parts[0] = len(pconfig['map'])
        else:
            self.manual = True
            for k, v in pconfig['partition'].items():
                self.parts[k] = v
        self.parts = tuple(self.parts)
        self.mapping = pconfig['map']
        mesh_size = list(self.parts) + [-1]
        self.mesh = pconfig['mesh'].reshape(mesh_size)
        self.strides = get_strides(self.parts)

    @property
    def is_replicate(self):
        assert False
        return self.mapping[0] == MapType.REPLICATE

    @property
    def is_shard(self):
        return len(self.mapping) == self.mesh.size

    @property
    def replicate_degree(self):
        assert self.mesh.ndim == len(self.parts) + 1
        return self.mesh.shape[-1]

    def random(self):
        assert False
        # partition
        # self.parts = tuple([r.random() for r in self.ranges])
        self.parts = tuple(
            [self.ndevs if i == 0 else 1 for i, r in enumerate(self.ranges)])
        self.strides = get_strides(self.parts)
        # device mapping
        if np.prod(self.parts) == 1:
            map_num = random.randint(1, self.ndevs)
            map_devs = random.sample(range(self.ndevs), map_num)
            self.mapping = (MapType.REPLICATE, tuple(map_devs))
        else:
            map_devs = random.choices(range(self.ndevs), k=np.prod(self.parts))
            self.mapping = (MapType.SHARD, tuple(map_devs))
        return self.parts, self.mapping

    @property
    def dev_type(self):
        if len(self.mapping) == 0:
            return None
        dev_type = self.mapping[0][1].split(':')[0]
        for dev in self.mapping:
            assert dev_type == dev[1].split(':')[0]
        return dev_type
