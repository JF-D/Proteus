import weakref
import numpy as np
from collections import OrderedDict, defaultdict
from proteus.type import OpType, MapType
from proteus.utils import divide_into_n_parts, flat_to_coordinate
from proteus.utils.utils import get_strides
from .node_config import TensorConfig, OpConfig
from .tensor import Input, Parameter, Buffer


def has_weight(op):
    if op.type in [
            OpType.Linear, OpType.Conv2d, OpType.LayerNorm, OpType.BatchNorm2d,
            OpType.Embedding
    ] and len(op.ins) > 1:
        return True
    return False


class STreeNode:
    _id_map = {}

    def __init__(self,
                 name,
                 type,
                 ins=None,
                 parent=None,
                 vgroup=False,
                 is_root=False):
        self.name = name
        self.type = type
        if ins is not None:
            self.ins = tuple(ins) if isinstance(ins,
                                                (list, tuple)) else (ins, )
        else:
            self.ins = ()
        self.outs = ()
        self.op = None

        self.parent = parent
        self.children = OrderedDict()

        self.vgroup = vgroup
        self.is_root = is_root
        if self.is_root:
            self.old_children = None
        self.pp_splited = False
        self.dev_mesh = defaultdict(lambda: None)
        self.pconfig = {}

        self.manual_splited = False
        self.split_info = defaultdict(dict)
        self.schedule_info = {}
        self.replica = False
        self.recomputation = False

    def set_ins(self, ins):
        self.ins = tuple(ins) if isinstance(ins, (list, tuple)) else (ins, )

    def set_outs(self, outs=None):
        if outs is None:
            return
        self.outs = tuple(outs) if isinstance(outs,
                                              (list, tuple)) else (outs, )

    def set_op(self, op):
        self.op = op
        STreeNode._id_map[op.id] = self

    def add_child(self, name, type, ins):
        child = STreeNode(name, type, ins, parent=self)
        if name in self.children:
            self.children[name + str(len(self.children))] = child
        else:
            self.children[name] = child
        return child

    def init_config(self, ndevs, max_parts=None, stride=1):
        self.opconfig = OpConfig(self.op,
                                 ndevs,
                                 max_parts=max_parts,
                                 stride=stride)
        self.data_config = {}
        for data in self.op.read:
            if not isinstance(data, (Input, Parameter, Buffer)):
                continue
            if self.op.type in [OpType.SGDApply, OpType.AdamApply]:
                if not isinstance(data, Buffer):
                    continue
            assert len(data.producer) == 0
            self.data_config[data.id] = TensorConfig(data,
                                                     ndevs,
                                                     max_parts=max_parts,
                                                     stride=stride)
        for data in self.op.write:
            self.data_config[data.id] = TensorConfig(data,
                                                     ndevs,
                                                     max_parts=max_parts,
                                                     stride=stride)

        return self.opconfig, self.data_config

    def init_optimizer(self):
        opts = []
        if len(self.children) == 0:
            for data in self.op.read:
                if isinstance(data, Parameter):
                    assert data.optimizer is not None
                    if data.optimizer not in opts:
                        opts.append(STreeNode._id_map[data.optimizer.id])
        else:
            for name, node in self.children.items():
                ret = node.init_optimizer()
                opts.extend(ret)
        self.optimizer_ops = opts
        return opts

    def split(self, dims, parts, item='op'):
        if isinstance(dims, int):
            self.split_info[item][dims] = parts
        else:
            for k, v in zip(dims, parts):
                self.split_info[item][k] = v

    def map(self, dev_mesh, item='op'):
        self.dev_mesh[item] = dev_mesh

    def recompute(self, recomputation=True):
        self.recomputation = recomputation
        self.schedule_info['recomputation'] = recomputation
        self.schedule_info['exclude'] = self.outs

    def propagate(self, graph):
        if len(self.children) > 0:
            for k, child in self.children.items():
                for key, dev_mesh in self.dev_mesh.items():
                    if child.dev_mesh[key] is None:
                        child.map(dev_mesh, item=key)
                for key, split_dict in self.split_info.items():
                    if len(child.split_info[key]) == 0:
                        dims, parts = [], []
                        for k, v in split_dict.items():
                            dims.append(k)
                            parts.append(v)
                        child.split(dims, parts, item=key)
                for key, value in self.schedule_info.items():
                    child.schedule_info[key] = value
                child.propagate(graph)
            return

        if len(self.dev_mesh) == 0:
            return

        pconfig = {'partition': {}, 'data': {}}
        # for key in [
        #         'op', 'weight', 'bias', 'weight_grad', 'bias_grad', 'out',
        #         'out_grad'
        # ]:
        keys = list(self.split_info.keys()) + list(self.dev_mesh.keys())
        for key in keys:
            split_dict = self.split_info[key]
            if len(split_dict) == 0 and self.dev_mesh[key] is None:
                continue
            if key == 'op':
                deg = 1
                parts = []
                for k in sorted(split_dict.keys()):
                    v = split_dict[k]
                    pconfig['partition'][k] = v
                    deg = deg * v
                    parts.append(v)

                mapdevs = self.dev_mesh[key].reshape(-1).tolist()
                if deg == 1 and len(mapdevs) > 1:
                    mapping = [tuple([MapType.REPLICATE] + mapdevs)]
                elif len(mapdevs) == deg:
                    mapping = []
                    for i in range(deg):
                        mapping.append((MapType.SHARD, mapdevs[i]))
                else:
                    strides = get_strides(parts)
                    mapping = []
                    for flat_idx in range(deg):
                        coord = flat_to_coordinate(flat_idx, strides)
                        mapdevs = self.dev_mesh[key][coord].reshape(
                            -1).tolist()
                        if len(mapdevs) > 1:
                            mapping.append(tuple([MapType.REPLICATE] +
                                                 mapdevs))
                        elif len(mapdevs) == 1:
                            mapping.append((MapType.SHARD, mapdevs[0]))
                        else:
                            raise NotImplementedError
                pconfig['map'] = mapping
                pconfig['mesh'] = self.dev_mesh[key]
            else:
                if key.startswith('weight') and not has_weight(self.op):
                    continue
                pconfig['data'][key] = {'partition': {}}
                deg = 1
                parts = []
                for k in sorted(split_dict.keys()):
                    v = split_dict[k]
                    pconfig['data'][key]['partition'][k] = v
                    deg = deg * v
                    parts.append(v)

                mapdevs = self.dev_mesh[key].reshape(-1).tolist()
                if deg == 1 and len(mapdevs) > 1:
                    mapping = [tuple([MapType.REPLICATE] + mapdevs)]
                elif len(mapdevs) == deg:
                    mapping = []
                    for i in range(deg):
                        mapping.append((MapType.SHARD, mapdevs[i]))
                else:
                    strides = get_strides(parts)
                    mapping = []
                    for flat_idx in range(deg):
                        coord = flat_to_coordinate(flat_idx, strides)
                        mapdevs = self.dev_mesh[key][coord].reshape(
                            -1).tolist()
                        if len(mapdevs) > 1:
                            mapping.append(tuple([MapType.REPLICATE] +
                                                 mapdevs))
                        elif len(mapdevs) == 1:
                            mapping.append((MapType.SHARD, mapdevs[0]))
                        else:
                            raise NotImplementedError
                pconfig['data'][key]['map'] = mapping
                pconfig['data'][key]['mesh'] = self.dev_mesh[key]

        if 'recomputation' in self.schedule_info and self.schedule_info['recomputation']:
            if self.outs[0] not in self.schedule_info['exclude']:
                if 'out' not in pconfig['data']:
                    pconfig['data']['out'] = {}
                pconfig['data']['out']['recompute'] = True
                self.recomputation = True

        self.pconfig = pconfig

        ops = [self.op]
        if self.op.id in graph.fwop_map:
            ops.append(graph.ops[graph.fwop_map[self.op.id]])
        graph.set_op_pconfig(ops, self.pconfig)

    def parameters(self):
        if len(self.children) == 0:
            for data in self.op.read:
                if isinstance(data, Parameter):
                    yield data
            return
        for _, child in self.children.items():
            for param in child.parameters():
                yield param

    def ops(self):
        if len(self.children) == 0:
            yield self.op
        else:
            for _, child in self.children.items():
                for op in child.ops():
                    yield op

    def __getattr__(self, name):
        if self.is_root and self.old_children:
            if name in self.old_children.keys():
                return self.old_children[name]
        if name in self.children.keys():
            return self.children[name]
        raise AttributeError

    def __repr__(self):
        string = '{}, {}, {}, {}, {}\n'.format(self.name, self.type, self.op,
                                               self.ins, self.outs)

        def dump_child(cur_node, nspace):
            nonlocal string
            for name, child in cur_node.children.items():
                string += '{}{}, {}, {}, {}, {}\n'.format(
                    ' ' * nspace, child.name, child.type, child.op, child.ins,
                    child.outs)
                dump_child(child, nspace + 2)

        dump_child(self, 2)
        return string


class StrategyTree:

    def __init__(self, name=None, type=None, ins=None):
        self.root = None
        self.optimizer = None
        if name is None and type is None:
            self.is_initialized = False
        else:
            self.make_root(name, type, ins)

        self.leaf_nodes = {}
        self.__refs = defaultdict(list)

        self.collective_comm = True
        self.bucket_size = 25
        self.overlap_grad_comm = True

    def make_root(self, name, type, ins):
        self.root = STreeNode(name, type, ins, is_root=True)
        self.is_initialized = True

    def make_optimizer(self, name, type, ins):
        self.optimizer = STreeNode(name, type, ins)

    def disable_collective_comm(self):
        self.collective_comm = False

    def disable_gradient_overlap(self):
        self.overlap_grad_comm = False

    def set_bucket_size(self, bucket_size=25):
        self.bucket_size = bucket_size

    def add_leaf_node(self, id, node):
        self.leaf_nodes[id] = node

    def add_instance(self, type, node):
        instances = self.__refs[type]
        instances.append(weakref.ref(node, lambda ref: instances.remove(ref)))

    def get_instances(self, cls):
        return self.__refs[cls]

    @property
    def refs(self):
        return self.__refs

    def init_config(self, dev_topo, max_parts=None, stride=1):
        self.dev_topo = dev_topo
        self.max_parts = max_parts
        self.stride = stride
        self._op_config = {}
        self._data_config = {}
        for k in self.leaf_nodes:
            opcfg, data_cfg = self.leaf_nodes[k].init_config(
                dev_topo.ndevs, max_parts=max_parts, stride=stride)
            assert opcfg.id not in self._op_config
            self._op_config[opcfg.id] = k

            tb_poped = []
            for idx in data_cfg:
                if idx in self._data_config:
                    tb_poped.append(idx)
                    continue
                self._data_config[idx] = k
            for idx in tb_poped:
                data_cfg.pop(idx)

    def op_config(self, op_id):
        node_id = self._op_config[op_id]
        return self.leaf_nodes[node_id].opconfig

    def data_config(self, data_id):
        node_id = self._data_config[data_id]
        return self.leaf_nodes[node_id].data_config[data_id]

    def schedule(self,
                 n_macro_batch=1,
                 interleave_freq=1,
                 max_ongoing_macro_batch=1,
                 nstages=None):
        self.root.pconfig['schedule'] = {
            'n_macro_batch': n_macro_batch,
            'interleave_freq': interleave_freq,
            'max_ongoing_macro_batch': max_ongoing_macro_batch
        }

        stages, meshes = defaultdict(list), []
        stage_id, stage_mesh = 0, None
        for i, (name, node) in enumerate(self.root.children.items()):
            if nstages and nstages[0] == 1:
                stages[stage_id].append(i)
                meshes.append(nstages[1])
                continue
            if stage_mesh is None:
                stages[stage_id].append(i)
                stage_mesh = node.dev_mesh['op']
                meshes.append(node.dev_mesh['op'])
            else:
                if node.dev_mesh['op'] is None or stage_mesh.tolist(
                ) == node.dev_mesh['op'].tolist():
                    stages[stage_id].append(i)
                else:
                    stage_id += 1
                    stages[stage_id].append(i)
                    stage_mesh = node.dev_mesh['op']
                    meshes.append(node.dev_mesh['op'])
        keys = list(self.root.children.keys())
        ret = []
        new_children = OrderedDict()
        for i in range(len(stages)):
            vnode = STreeNode(str(i), 'VNode', parent=self.root, vgroup=True)
            vnode.map(meshes[i])
            # deal with children
            vins, vouts, produced = OrderedDict(), OrderedDict(), set()
            for layer_id in stages[i]:
                k = keys[layer_id]
                for data in self.root.children[k].ins:
                    if isinstance(data, (tuple, list)):
                        for d in data:
                            if d.id not in vouts:
                                if d.id not in produced:
                                    vins[d.id] = d
                            else:
                                vouts.pop(d.id)
                                produced.add(d.id)
                    else:
                        if data.id not in vouts:
                            if data.id not in produced:
                                vins[data.id] = data
                        else:
                            vouts.pop(data.id)
                            produced.add(data.id)
                for data in self.root.children[k].outs:
                    vouts[data.id] = [data, len(data.consumer)]

                self.root.children[k].parent = vnode
                vnode.children[k] = self.root.children[k]
            for param in vnode.parameters():
                vins[param.id] = param
            vnode.set_ins([vins[data_id] for data_id in vins])
            vnode.set_outs([vouts[data_id][0] for data_id in vouts])
            new_children[f'vnode_{i}'] = vnode
            ret.append(vnode)

        self.root.old_children = self.root.children
        self.root.children = new_children
        self.root.pp_splited = True

        for i, (name, node) in enumerate(self.root.children.items()):
            if node.type != 'VNode':
                continue
            if 'schedule' not in node.pconfig:
                node.pconfig['schedule'] = {}
            if 'n_macro_batch' not in node.pconfig['schedule']:
                node.pconfig['schedule']['n_macro_batch'] = n_macro_batch
            if 'interleave_freq' not in node.pconfig['schedule']:
                node.pconfig['schedule']['interleave_freq'] = interleave_freq
            if 'max_ongoing_macro_batch' not in node.pconfig['schedule']:
                if isinstance(max_ongoing_macro_batch, int):
                    max_ongo = max_ongoing_macro_batch
                elif isinstance(max_ongoing_macro_batch, (list, tuple)):
                    max_ongo = max_ongoing_macro_batch[i]
                node.pconfig['schedule']['max_ongoing_macro_batch'] = max_ongo
        return ret

    def propagate(self, graph):
        self.root.propagate(graph)
        self.optimizer.propagate(graph)

    def __getattr__(self, name):
        if name in self.root.children.keys():
            if self.root.pp_splited:
                if name in self.root.children:
                    return self.root.children[name]
                for vnode in self.root.children.values():
                    if vnode.type == 'VNode' and name in vnode.children:
                        return vnode.children[name]
                # raise error
                return self.root.children[name]
            else:
                return self.root.children[name]

        return getattr(self.root, name)

    def dump_tree(self, config=False):
        print(self.root.name, self.root.type, self.root.op, self.root.ins,
              self.root.outs, f'  [{self.root.recomputation}]')
        if config:
            print(self.root.recomputation)

        def dump_child(cur_node, nspace):
            for name, child in cur_node.children.items():
                print(' ' * nspace, child.name, child.type, child.op,
                      child.ins, child.outs, f'  [{child.recomputation}]')
                if config:
                    print(' ' * nspace, child.recomputation)
                dump_child(child, nspace + 2)

        dump_child(self.root, 2)
        if self.optimizer is not None:
            print(self.optimizer.name, self.optimizer.type, self.optimizer.op,
                  self.optimizer.ins, self.optimizer.outs, f'  [{self.optimizer.recomputation}]')
            if config:
                print(self.optimizer.recomputation)
            dump_child(self.optimizer, 2)
