import random
import numpy as np
from collections import namedtuple
from proteus import DataType, MapType, IterType
from proteus.simulator.cost_model import OpCostModel
from proteus.type import OpType
from proteus.utils import get_strides, flat_to_coordinate, coordinate_to_flat
from proteus.utils.interval_analysis import Interval, InputSpace, get_coordinate_interval, get_iters_interval


class Node(object):

    def __init__(self):
        super().__init__()


class Op(Node):
    id = 0

    def __init__(self, ins, outs, attr=None, name=''):
        super().__init__()
        self.read = tuple(ins)
        self.write = tuple(outs)
        self.attr = attr
        self.name = name

        for idx, data in enumerate(ins):
            data.add_consumer(self, idx)
        for idx, data in enumerate(outs):
            data.add_producer(self, idx)

        self.partitioned = False

        self.prefetch_from_cpu = {}
        self.prefetch_from_gpu = {}

        self.depth = None
        self.id = Op.id
        Op.id += 1

    def get_input_space(self):
        self.input_space = InputSpace.get_identity(self.iter_space)
        return self.input_space

    def add_prefetch_task(self, task, rid, group, from_dev='gpu'):
        prefetch_dict = self.prefetch_from_gpu if from_dev == 'gpu' else self.prefetch_from_cpu
        if rid not in prefetch_dict:
            prefetch_dict[rid] = {}
        for dev in group:
            if dev not in prefetch_dict[rid]:
                prefetch_dict[rid][dev] = []
            prefetch_dict[rid][dev].append(task)

    def partition(self, config):
        parts, mapping = tuple(config.parts), tuple(config.mapping)
        if self.partitioned:
            if (parts, mapping) == (self._parts, self._mapping):
                return
        self.partitioned = True
        self._parts = parts
        self._mapping = mapping

        self.sub_outs, self.sub_ins = {}, {}
        self.get_input_space()

        if IterType.CONTINUOUS in self.iter_space.iters:
            in_parts = [
                self.get_in_part(0, i, config) for i in range(len(self.ins[0]))
            ]
            out_parts = [
                self.get_out_part(0, i, config)
                for i in range(len(self.outs[0]))
            ]
            assert np.prod(in_parts) == np.prod(out_parts)
            for i in range(np.prod(in_parts)):
                in_interval = get_iters_interval(i, in_parts, self.ins[0])
                out_interval = get_iters_interval(i, out_parts, self.outs[0])
                self.sub_outs[i] = (out_interval, )
                self.sub_ins[i] = (in_interval, )
        else:
            in_parts = [
                self.get_input_parts(i, config.parts)
                for i in range(len(self.ins))
            ]
            out_parts = [
                self.get_output_parts(i, config.parts)
                for i in range(len(self.outs))
            ]
            for i in range(config.deg()):
                coordinate = flat_to_coordinate(i, config.strides)
                sub_outs, sub_ins = [], []
                for k, out_iter in enumerate(self.iter_space.out_iters):
                    sub_coord = [coordinate[oid] for oid in out_iter]
                    sub_outs.append(
                        get_coordinate_interval(sub_coord, out_parts[k],
                                                self.outs[k]))

                for k, in_iter in enumerate(self.iter_space.in_iters):
                    sub_coord = [coordinate[iid] for iid in in_iter]
                    sub_ins.append(
                        get_coordinate_interval(sub_coord, in_parts[k],
                                                self.ins[k]))
                self.sub_outs[i] = tuple(sub_outs)
                self.sub_ins[i] = tuple(sub_ins)

    def get_in_part(self, in_id, dim, config):
        dim = len(self.ins[in_id]) + dim if dim < 0 else dim
        in_iter = self.iter_space.in_iters[in_id]
        part = config.get_part(in_iter[dim])
        if self.iter_space.iters[in_iter[dim]] != IterType.CONTINUOUS:
            return part

        if len(self.ins[in_id]) == len(config.parts):
            return part
        else:
            part = 1
            for i, iter_type in enumerate(self.iter_space.iters):
                if iter_type == IterType.CONTINUOUS:
                    part = part * config.parts[i]
        return part

    def get_out_part(self, out_id, dim, config):
        dim = len(self.outs[out_id]) + dim if dim < 0 else dim
        out_iter = self.iter_space.out_iters[out_id]
        part = config.get_part(out_iter[dim])
        if self.iter_space.iters[out_iter[dim]] != IterType.CONTINUOUS:
            return part

        if len(self.outs[out_id]) == len(config.parts):
            return part
        else:
            part = 1
            for i, iter_type in enumerate(self.iter_space.iters):
                if iter_type == IterType.CONTINUOUS:
                    part = part * config.parts[i]
        return part

    def _get_in_part(self, in_id, dim, part, parts):
        dim = len(self.ins[in_id]) + dim if dim < 0 else dim
        in_iter = self.iter_space.in_iters[in_id]
        if self.iter_space.iters[in_iter[dim]] != IterType.CONTINUOUS:
            return part

        if len(self.ins[in_id]) == len(parts):
            return part
        else:
            part = 1
            for i, iter_type in enumerate(self.iter_space.iters):
                if iter_type == IterType.CONTINUOUS:
                    part = part * parts[i]
        return part

    def _get_out_part(self, out_id, dim, part, parts):
        dim = len(self.outs[out_id]) + dim if dim < 0 else dim
        out_iter = self.iter_space.out_iters[out_id]
        if self.iter_space.iters[out_iter[dim]] != IterType.CONTINUOUS:
            return part

        if len(self.outs[out_id]) == len(parts):
            return part
        else:
            part = 1
            for i, iter_type in enumerate(self.iter_space.iters):
                if iter_type == IterType.CONTINUOUS:
                    part = part * parts[i]
        return part

    def get_input_parts(self, in_id, parts):
        p = [
            self._get_in_part(in_id, i, parts[it], parts)
            for i, it in enumerate(self.iter_space.in_iters[in_id])
        ]
        return p

    def get_output_parts(self, out_id, parts):
        p = [
            self._get_out_part(out_id, i, parts[it], parts)
            for i, it in enumerate(self.iter_space.out_iters[out_id])
        ]
        return p

    def infer_input_config(self, ind, config):
        '''
        config: op partition config
        ind: the ind-th input to be infered
        '''
        parts = [
            self.get_in_part(ind, i, config) for i in range(len(self.ins[ind]))
        ]
        mapping = []
        strides = get_strides(parts)
        if IterType.CONTINUOUS in self.iter_space.iters:
            mesh_size = list(parts) + [-1]
            data_mesh = config.mesh.reshape(mesh_size)
        else:
            indice = list(self.iter_space.in_iters[ind])
            for i in range(config.mesh.ndim):
                if i not in indice:
                    indice.append(i)
            data_mesh = config.mesh.transpose(indice)
            mesh_shape = data_mesh.shape[:len(self.iter_space.in_iters[ind])]
            data_mesh = data_mesh.reshape(list(mesh_shape) + [-1])

        for flat_index in range(np.prod(parts)):
            coordinate = flat_to_coordinate(flat_index, strides)
            map_devs = data_mesh[coordinate].reshape(-1)
            if map_devs.size == 1:
                mapping.append((MapType.SHARD, map_devs[0]))
            else:
                mapping.append(tuple([MapType.REPLICATE] + list(map_devs)))
        return tuple(parts), mapping, data_mesh

    def infer_output_config(self, ind, config):
        '''
        config: op partition config
        ind: the ind-th output to be infered
        '''
        parts = [
            self.get_out_part(ind, i, config)
            for i in range(len(self.outs[ind]))
        ]
        mapping = []
        strides = get_strides(parts)
        if IterType.CONTINUOUS in self.iter_space.iters:
            mesh_size = list(parts) + [-1]
            data_mesh = config.mesh.reshape(mesh_size)
        else:
            indice = list(self.iter_space.out_iters[ind])
            for i in range(config.mesh.ndim):
                if i not in indice:
                    indice.append(i)
            data_mesh = config.mesh.transpose(indice)
            mesh_shape = data_mesh.shape[:len(self.iter_space.out_iters[ind])]
            data_mesh = data_mesh.reshape(list(mesh_shape) + [-1])
        for flat_index in range(np.prod(parts)):
            coordinate = flat_to_coordinate(flat_index, strides)
            map_devs = data_mesh[coordinate].reshape(-1)
            if map_devs.size == 1:
                mapping.append((MapType.SHARD, map_devs[0]))
            else:
                mapping.append(tuple([MapType.REPLICATE] + list(map_devs)))
        return tuple(parts), mapping, data_mesh

    def set_input_config(self, idx, cfg):
        if 'in_cfg' not in self.__dict__:
            self.in_cfg = {}
        self.in_cfg[idx] = cfg.get_config()

    def set_output_config(self, idx, cfg):
        if 'out_cfg' not in self.__dict__:
            self.out_cfg = {}
        self.out_cfg[idx] = cfg.get_config()

    def infer_op_from_input(self):
        if 'in_cfg' not in self.__dict__:
            return None
        iters, data_cfgs = [], []
        for idx, cfg in self.in_cfg.items():
            iters.append(self.iter_space.in_iters[idx])
            data_cfgs.append(cfg)
        parts = self.check_tensor_config_compatible(iters, data_cfgs)
        if OpType.is_optimizer(self.type) and np.prod(parts) == 1:
            replica = True
            mapdevs = []
            for cfg in data_cfgs:
                if np.prod(cfg[0]) != 1 or cfg[1][0][0] != MapType.REPLICATE:
                    replica = False
                    break
                mapdevs.extend(cfg[1][0][1:])
            if replica:
                mapping = (tuple([MapType.REPLICATE] + mapdevs), )
                mesh = np.array(mapdevs).reshape(parts + [-1])
                return parts, mapping, mesh
        mapdevs = []
        if parts:
            strides = get_strides(parts)
            dt_strides = [get_strides(cfg[0]) for cfg in data_cfgs]
            mapping, mapdevs = [], []

            for flat_idx in range(np.prod(parts)):
                mdev, backup = [], []
                coord = flat_to_coordinate(flat_idx, strides)
                for indice, cfg, substr in zip(iters, data_cfgs, dt_strides):
                    if IterType.CONTINUOUS in self.iter_space.iters:
                        in_parts = self.get_input_parts(0, parts)
                        sub_coord = flat_to_coordinate(flat_idx, get_strides(in_parts))
                    else:
                        sub_coord = [coord[i] for i in indice]
                    sub_flat = coordinate_to_flat(sub_coord, substr)
                    if sub_flat >= np.prod(cfg[0]):
                        backup.append(cfg[1][sub_flat % np.prod(cfg[0])][1])
                    else:
                        # deal with replicated config
                        if len(mdev) == 0:
                            mdev = list(cfg[1][sub_flat][1:])
                        else:
                            new_mdev = []
                            for dev_id in cfg[1][sub_flat][1:]:
                                if dev_id in mdev:
                                    new_mdev.append(dev_id)
                            mdev = new_mdev
                assert len(mdev) > 0
                if len(mdev) == 1:
                    mapping.append((MapType.SHARD, mdev[0]))
                else:
                    mapping.append(tuple([MapType.REPLICATE] + mdev))
                mapdevs += mdev
            mapping = tuple(mapping)
            assert np.prod(parts) == len(mapping)
            mesh = np.array(mapdevs).reshape(list(parts) + [-1])
            return parts, mapping, mesh

        return None

    def infer_op_from_output(self):
        if 'out_cfg' not in self.__dict__:
            return None
        iters, data_cfgs = [], []
        for idx, cfg in self.out_cfg.items():
            iters.append(self.iter_space.out_iters[idx])
            data_cfgs.append(cfg)
        parts = self.check_tensor_config_compatible(iters, data_cfgs)
        mapdevs = []
        if parts:
            strides = get_strides(parts)
            dt_strides = [get_strides(cfg[0]) for cfg in data_cfgs]
            mapping, mapdevs = [], []
            for flat_idx in range(np.prod(parts)):
                mdev, backup = [], []
                coord = flat_to_coordinate(flat_idx, strides)
                for indice, cfg, substr in zip(iters, data_cfgs, dt_strides):
                    if IterType.CONTINUOUS in self.iter_space.iters:
                        out_parts = self.get_output_parts(0, parts)
                        sub_coord = flat_to_coordinate(flat_idx,
                                                       get_strides(out_parts))
                    else:
                        sub_coord = [coord[i] for i in indice]
                    sub_flat = coordinate_to_flat(sub_coord, substr)
                    if sub_flat >= np.prod(cfg[0]):
                        backup.append(cfg[1][sub_flat % np.prod(cfg[0])][1])
                    else:
                        if len(mdev) == 0:
                            mdev = list(cfg[1][sub_flat][1:])
                        else:
                            new_mdev = []
                            for dev_id in cfg[1][sub_flat][1:]:
                                if dev_id in mdev:
                                    new_mdev.append(dev_id)
                assert len(mdev) > 0
                if len(mdev) == 1:
                    mapping.append((MapType.SHARD, mdev[0]))
                else:
                    mapping.append(tuple([MapType.REPLICATE] + mdev))
                mapdevs += mdev
            mapping = tuple(mapping)
            assert np.prod(parts) == len(mapping)
            mesh = np.array(mapdevs).reshape(list(parts) + [-1])
            return parts, mapping, mesh

        return None

    def check_tensor_config_compatible(self, iters, data_cfgs):
        # a strict check
        parts = [1] * len(self.iter_space.iters)
        if IterType.CONTINUOUS in self.iter_space.iters:
            assert len(iters) == 1
            for indices, cfg in zip(iters, data_cfgs):
                if len(indices) == len(self.iter_space.iters):
                    parts = list(cfg[0])
                    return parts
                else:
                    for i, p in zip(indices, cfg[0]):
                        if self.iter_space.iters[i] != IterType.CONTINUOUS or p == 1:
                            parts[i] = p
                        else:
                            return []
            return parts

        for indice, cfg in zip(iters, data_cfgs):
            for i, p in zip(indice, cfg[0]):
                if parts[i] == 1:
                    parts[i] = p
                elif parts[i] != p:
                    return []
        return parts

    def measure_cost(self,
                     dev,
                     cost_type='roofline',
                     reprofile=False,
                     profile_iters=10):
        self.cost = OpCostModel.measure_op_cost(self,
                                                dev,
                                                cost_type,
                                                reprofile=reprofile,
                                                profile_iters=profile_iters)
        return self.cost

    def __repr__(self):
        string = str(self.type) + '_' + str(self.id)
        return string


Edge = namedtuple('Edge', ['privilege', 'op', 'id', 'data'])
