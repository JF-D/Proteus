from . import optim

_DEFAULT_GRAPH = None
_DEFAULT_STRATEGY_TREE = None
_DEFAULT_STRATEGY_NODE = None


def register_op(type, ins, *args, name='', **kwargs):
    global _DEFAULT_GRAPH, _DEFAULT_STRATEGY_NODE
    if _DEFAULT_GRAPH is None:
        from .graph import ProteusModel
        _DEFAULT_GRAPH = ProteusModel()
    scope_name = name.split('.')[-1]
    if scope_name == '':
        scope_name = 'seq' + str(len(_DEFAULT_STRATEGY_NODE.children))
    enter_name_scope(scope_name, type, ins)

    outs = getattr(_DEFAULT_GRAPH, type)(*ins, *args, name=name, **kwargs)

    if isinstance(outs, (tuple, list)):
        op = outs[0].producer[-1][0]
    else:
        op = outs.producer[-1][0]
    _DEFAULT_STRATEGY_NODE.set_op(op)
    _DEFAULT_STRATEGY_TREE.add_leaf_node(op.id, _DEFAULT_STRATEGY_NODE)

    exit_name_scope(outs)
    return outs


def register_placeholder(type, *args, name='', **kwargs):
    global _DEFAULT_GRAPH, _DEFAULT_STRATEGY_NODE
    if _DEFAULT_GRAPH is None:
        from .graph import ProteusModel
        _DEFAULT_GRAPH = ProteusModel()
    outs = getattr(_DEFAULT_GRAPH, type)(*args, name=name, **kwargs)
    return outs


def register_optimizer(type, param, **kwargs):
    opt = getattr(optim, type)(_DEFAULT_GRAPH, param, **kwargs)
    enter_name_scope('', type, opt.op.read)
    _DEFAULT_STRATEGY_NODE.set_op(opt.op)
    _DEFAULT_STRATEGY_TREE.add_leaf_node(opt.op.id, _DEFAULT_STRATEGY_NODE)
    exit_name_scope(None)


def enter_name_scope(name, type, ins):
    global _DEFAULT_STRATEGY_TREE, _DEFAULT_STRATEGY_NODE
    scope_name = name if name else type.lower()
    if _DEFAULT_STRATEGY_TREE is None:
        from .strategy_tree import StrategyTree
        _DEFAULT_STRATEGY_TREE = StrategyTree(scope_name, type, ins)
        _DEFAULT_STRATEGY_NODE = _DEFAULT_STRATEGY_TREE.root
        return
    if not _DEFAULT_STRATEGY_TREE.is_initialized:
        _DEFAULT_STRATEGY_TREE.make_root(scope_name, type, ins)
        _DEFAULT_STRATEGY_NODE = _DEFAULT_STRATEGY_TREE.root
        return
    if name == 'optimizer':
        if _DEFAULT_STRATEGY_TREE.optimizer is None:
            _DEFAULT_STRATEGY_TREE.make_optimizer(name, type, ins)
        _DEFAULT_STRATEGY_NODE = _DEFAULT_STRATEGY_TREE.optimizer
    else:
        _DEFAULT_STRATEGY_NODE = _DEFAULT_STRATEGY_NODE.add_child(
            scope_name, type, ins)
    # record instance
    _DEFAULT_STRATEGY_TREE.add_instance(type, _DEFAULT_STRATEGY_NODE)


def exit_name_scope(outs):
    global _DEFAULT_STRATEGY_NODE
    _DEFAULT_STRATEGY_NODE.set_outs(outs)
    if _DEFAULT_STRATEGY_NODE.parent is not None:
        _DEFAULT_STRATEGY_NODE = _DEFAULT_STRATEGY_NODE.parent


def get_default_graph():
    return _DEFAULT_GRAPH, _DEFAULT_STRATEGY_TREE


class Context:
    def __init__(self, graph, stree):
        self.graph = graph
        self.stree = stree

    def __enter__(self):
        global _DEFAULT_GRAPH, _DEFAULT_STRATEGY_TREE, _DEFAULT_STRATEGY_NODE
        self.old_graph = _DEFAULT_GRAPH
        self.old_stree = _DEFAULT_STRATEGY_TREE
        self.old_node = _DEFAULT_STRATEGY_NODE

        _DEFAULT_GRAPH = self.graph
        _DEFAULT_STRATEGY_TREE = self.stree
        _DEFAULT_STRATEGY_NODE = self.stree.root

    def __exit__(self, *args):
        global _DEFAULT_GRAPH, _DEFAULT_STRATEGY_TREE, _DEFAULT_STRATEGY_NODE
        _DEFAULT_GRAPH = self.old_graph
        _DEFAULT_STRATEGY_TREE = self.old_stree
        _DEFAULT_STRATEGY_NODE = self.old_node


def compile(model, inputs, criterion=None, optimizer=None):
    global _DEFAULT_GRAPH, _DEFAULT_STRATEGY_TREE
    from .graph import ProteusModel
    from .strategy_tree import StrategyTree
    graph = ProteusModel(train=model.training)
    stree = StrategyTree()

    with Context(graph, stree):
        datas, labels = [], []
        for idx, in_shape in enumerate(inputs['input']):
            datas.append(_DEFAULT_GRAPH.Placeholder(in_shape, name=f'x_{idx}'))
        for idx, label_shape in enumerate(inputs['label']):
            labels.append(
                _DEFAULT_GRAPH.Placeholder(label_shape, name=f'label_{idx}'))
        y = model(*datas)
        if criterion is not None:
            c_type = criterion.__class__.__name__
            if c_type == 'function':
                c_type = criterion.__name__
            ins = list(y) if isinstance(y, (tuple, list)) else [y]
            enter_name_scope('criterion', c_type, ins + list(labels))
            loss = criterion(y, *labels)
            exit_name_scope(loss)
            _DEFAULT_STRATEGY_TREE.root.set_outs(loss)
        if optimizer is not None:
            enter_name_scope('optimizer', optimizer.__class__.__name__, None)
            optimizer.step()
            exit_name_scope(None)
            ins = []
            for opt_op in _DEFAULT_STRATEGY_TREE.optimizer.ops():
                ins.extend(opt_op.read)
            _DEFAULT_STRATEGY_TREE.optimizer.set_ins(ins)
    stree.root.init_optimizer()
    return graph, stree
