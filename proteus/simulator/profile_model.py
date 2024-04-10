import time
import random
import numpy as np
import torch
import torch.nn as nn
from proteus.type import DevType, OpType
from .cost_model import OpCostModel, get_profile_iters, set_profile_iters

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True


def perf_wrap(func, *args, **kwargs):
    torch.cuda.synchronize()
    st = time.perf_counter()
    ret = func(*args, **kwargs)
    torch.cuda.synchronize()
    ed = time.perf_counter()
    return ret, (ed - st) * 1000


class ProfileWrapper(object):

    def __init__(self, func, warmup_iters=20):
        super().__init__()
        self.func = func
        self.warmup_iters = warmup_iters

    def profile(self):
        niters = get_profile_iters()
        torch.cuda.synchronize()
        st = time.perf_counter()
        for i in range(niters):
            self.func(i)
        torch.cuda.synchronize()
        ed = time.perf_counter()
        return (ed - st) * 1000 / niters

    def __enter__(self):
        for i in range(self.warmup_iters):
            self.func(i)
        return self

    def __exit__(self, type, value, traceback):
        pass


def _make_input_var(x):
    if x.dtype == torch.int64:
        out = torch.randint(0, torch.max(x), x.size(), dtype=x.dtype)
    elif x.dtype == torch.bool:
        out = torch.randn(x.size()) > 0.5
    else:
        out = torch.randn(x.size(), dtype=x.dtype)
    out = out.to(x.device)
    out.requires_grad = x.requires_grad
    return out


def _make_input_vars(x):
    if isinstance(x, torch.Tensor):
        return _make_input_var(x)
    elif isinstance(x, (tuple, list)):
        ret = []
        for data in x:
            ret.append(_make_input_vars(data))
        return ret


def make_input_vars(ins, n=3):
    outs = []
    for _ in range(n):
        if isinstance(ins, torch.Tensor):
            out = _make_input_vars(ins)
            outs.append(out)
        elif isinstance(ins, (list, tuple)):
            out = []
            for data in ins:
                out_ = _make_input_vars(data)
                out.append(out_)
            outs.append(tuple(out))
    return outs


def module_forward_helper(layer, dev, x, gen=True):
    if dev.type == DevType.GPU:
        layer.cuda()

    if gen:
        ins = make_input_vars(x)
    else:
        ins = []
        for _x in x:
            if isinstance(_x, (tuple, list)):
                t_l = []
                for t in _x:
                    t_l.append(t.cuda())
                ins.append(t_l)
            else:
                ins.append(_x.cuda())

    x = ins[0 % len(ins)]
    y = layer(*x)
    dy = torch.ones_like(y)

    def step(i):
        x = ins[i % len(ins)]
        y = layer(*x)
        return y

    with ProfileWrapper(step) as pfn:
        cost = pfn.profile()

    cost_list = []

    def step(i):
        if gen:
            x_ = _make_input_vars(x)
        else:
            x_ = ins[i % len(ins)]
        y, speed = perf_wrap(layer, *x_)
        y.backward(dy)
        cost_list.append(speed)

    with ProfileWrapper(step) as pfn:
        pfn.profile()

    cost = min(min(cost_list), cost)
    return cost


def module_backward_helper(layer, dev, x, gen=True):
    if dev.type == DevType.GPU:
        layer.cuda()

    if gen:
        ins = make_input_vars(x)
    else:
        ins = []
        for _x in x:
            if isinstance(_x, (tuple, list)):
                t_l = []
                for t in _x:
                    t_l.append(t.cuda())
                ins.append(t_l)
            else:
                ins.append(_x.cuda())

    with torch.no_grad():
        x = ins[0 % len(ins)]
        y = layer(*x)
        dy = torch.ones_like(y)

    def step_fw(i):
        x = ins[i % len(ins)]
        with torch.no_grad():
            y = layer(*x)

    def step(i):
        x = ins[i % len(ins)]
        y = layer(*x)
        y.backward(dy)

    with ProfileWrapper(step_fw) as pfn:
        cost_fw = pfn.profile()
    with ProfileWrapper(step) as pfn:
        cost = pfn.profile()

    cost_list_fw, cost_list_bw = [], []

    def step(i):
        if gen:
            x_ = _make_input_vars(x)
        else:
            x_ = ins[i % len(ins)]

        y, speed = perf_wrap(layer, *x_)
        cost_list_fw.append(speed)

        def backward():
            y.backward(dy)

        _, speed = perf_wrap(backward)
        cost_list_bw.append(speed)

    with ProfileWrapper(step) as pfn:
        pfn.profile()
    cost_bw = min(min(cost_list_bw), cost - cost_fw, cost - min(cost_list_fw))
    return cost_bw


def torch_forward_helper(fn_name, dev, xs, **kwargs):
    ins = make_input_vars(xs)

    def step(i):
        x = ins[i % len(ins)]
        y = getattr(torch, fn_name)(*x, **kwargs)

    with ProfileWrapper(step) as pfn:
        cost = pfn.profile()

    cost_list = []

    def step(i):
        x_ = _make_input_vars(xs)

        def forward(*x, **kwargs):
            y = getattr(torch, fn_name)(*x, **kwargs)
            return y

        y, speed = perf_wrap(forward, *x_, **kwargs)
        y.backward(torch.ones_like(y))
        cost_list.append(speed)

    with ProfileWrapper(step) as pfn:
        pfn.profile()
    cost = min(cost, min(cost_list))
    return cost


def torch_backward_helper(fn_name, dev, xs, **kwargs):
    ins = make_input_vars(xs)

    x = ins[0 % len(ins)]
    y = getattr(torch, fn_name)(*x, **kwargs)
    dy = torch.ones_like(y)

    def step_fw(i):
        x = ins[i % len(ins)]
        y = getattr(torch, fn_name)(*x, **kwargs)

    def step(i):
        x = ins[i % len(ins)]
        y = getattr(torch, fn_name)(*x, **kwargs)
        y.backward(dy)

    with ProfileWrapper(step_fw) as pfn:
        cost_fw = pfn.profile()
    with ProfileWrapper(step) as pfn:
        cost = pfn.profile()

    cost_list_fw, cost_list_bw = [], []

    def step(i):
        x_ = _make_input_vars(xs)

        def forward(*x, **kwargs):
            y = getattr(torch, fn_name)(*x, **kwargs)
            return y

        y, speed = perf_wrap(forward, *x_, **kwargs)
        cost_list_fw.append(speed)

        def backward():
            y.backward(dy)

        _, speed = perf_wrap(backward)
        cost_list_bw.append(speed)

    with ProfileWrapper(step) as pfn:
        pfn.profile()
    cost_bw = min(min(cost_list_bw), cost - cost_fw, cost - min(cost_list_fw))
    return cost_bw


def tensor_forward_helper(fn_name, dev, x, *args, **kwargs):
    ins = make_input_vars(x)

    def step(i):
        x = ins[i % len(ins)]
        y = getattr(x, fn_name)(*args, **kwargs)

    with ProfileWrapper(step) as pfn:
        cost = pfn.profile()
    return cost


def tensor_backward_helper(fn_name, dev, x, *args, **kwargs):
    ins = make_input_vars(x)

    x = ins[0 % len(ins)]
    y = getattr(x, fn_name)(*args, **kwargs)
    dy = torch.ones_like(y)

    def step_fw(i):
        x = ins[i % len(ins)]
        y = getattr(x, fn_name)(*args, **kwargs)

    def step(i):
        x = ins[i % len(ins)]
        y = getattr(x, fn_name)(*args, **kwargs)
        y.backward(dy)

    with ProfileWrapper(step_fw) as pfn:
        cost_fw = pfn.profile()
    with ProfileWrapper(step) as pfn:
        cost = pfn.profile()
    return cost - cost_fw


def function_forward_helper(fn, xs, *args, **kwargs):
    ins = make_input_vars(xs)

    def step(i):
        x = ins[i % len(ins)]
        y = fn(*x, *args, **kwargs)

    with ProfileWrapper(step) as pfn:
        cost = pfn.profile()

    cost_list = []

    def step(i):
        x_ = _make_input_vars(xs)

        def forward(*x, **kwargs):
            y = fn(*x, **kwargs)
            return y

        y, speed = perf_wrap(forward, *x_, *args, **kwargs)
        y.backward(torch.ones_like(y))
        cost_list.append(speed)

    with ProfileWrapper(step) as pfn:
        pfn.profile()
    cost = min(cost, min(cost_list))
    return cost


def function_backward_helper(fn, xs, *args, **kwargs):
    ins = make_input_vars(xs)

    x = ins[0 % len(ins)]
    y = fn(*x, *args, **kwargs)
    dy = torch.ones_like(y)

    def step_fw(i):
        x = ins[i % len(ins)]
        y = fn(*x, *args, **kwargs)

    def step(i):
        x = ins[i % len(ins)]
        y = fn(*x, *args, **kwargs)
        y.backward(dy)

    with ProfileWrapper(step_fw) as pfn:
        cost_fw = pfn.profile()
    with ProfileWrapper(step) as pfn:
        cost = pfn.profile()

    cost_list_fw, cost_list_bw = [], []

    def step(i):
        x_ = _make_input_vars(xs)

        def forward(*x, **kwargs):
            y = fn(*x, **kwargs)
            return y

        y, speed = perf_wrap(forward, *x_, *args, **kwargs)
        cost_list_fw.append(speed)

        def backward():
            y.backward(dy)

        _, speed = perf_wrap(backward)
        cost_list_bw.append(speed)

    with ProfileWrapper(step) as pfn:
        pfn.profile()
    cost_bw = min(min(cost_list_bw), cost - cost_fw, cost - min(cost_list_fw))
    return cost_bw


def optimizer_step(opt):

    def step(i):
        opt.step()

    old = get_profile_iters()
    set_profile_iters(1000)
    with ProfileWrapper(step) as pfn:
        cost = pfn.profile()
    set_profile_iters(old)
    return cost


def make_randn_tensor(size, dev_type, requires_grad=False):
    out = torch.randn(size)
    if dev_type == DevType.GPU:
        out = out.cuda()
    out.requires_grad = requires_grad
    return out


def make_randn_label(size, dev_type, nclasses):
    if size[0] == 1:
        out = torch.arange(size[1], dtype=torch.int64).reshape(size)
    else:
        out = torch.randint(0, nclasses, size, dtype=torch.int64)
    if dev_type == DevType.GPU:
        out = out.cuda()
    return out


def profile_linear(op, dev):
    x = make_randn_tensor(op.ins[0], dev.type)
    x.requires_grad = op.read[0].requires_grad
    layer = nn.Linear(op.ins[1][1], op.ins[1][0], bias=(len(op.ins) > 2))
    layer.train()
    return module_forward_helper(layer, dev, [x])


def profile_linear_bw(op, dev):
    x = make_randn_tensor(op.ins[1], dev.type)
    x.requires_grad = op.outs[0] == op.ins[1]
    layer = nn.Linear(op.ins[2][1], op.ins[2][0], bias=(len(op.outs[-1]) == 1))
    layer.train()
    return module_backward_helper(layer, dev, [x])


def profile_matmul(op, dev):
    assert len(op.ins) == 2
    in1 = make_randn_tensor(op.ins[0], dev.type)
    in2 = make_randn_tensor(op.ins[1], dev.type)
    in1.requires_grad = True
    in2.requires_grad = True
    return torch_forward_helper('matmul', dev, [in1, in2])


def profile_matmul_bw(op, dev):
    assert len(op.ins) == 3
    in1 = make_randn_tensor(op.ins[1], dev.type)
    in2 = make_randn_tensor(op.ins[2], dev.type)
    in1.requires_grad = True
    in2.requires_grad = True
    return torch_backward_helper('matmul', dev, [in1, in2])


def profile_conv2d(op, dev):
    x = make_randn_tensor(op.ins[0], dev.type)
    x.requires_grad = op.read[0].requires_grad
    layer = nn.Conv2d(op.ins[1][1],
                      op.ins[1][0],
                      op.ins[1][2:],
                      stride=op.attr['stride'],
                      padding=op.attr['padding'],
                      bias=(len(op.ins) > 2))
    layer.train()
    layer.cuda()
    for _ in range(50):
        y = layer(x)
        y.backward(torch.ones_like(y))
    return module_forward_helper(layer, dev, [x])


def profile_conv2d_bw(op, dev):
    x = make_randn_tensor(op.ins[1], dev.type)
    x.requires_grad = op.outs[0] == op.ins[1]
    layer = nn.Conv2d(op.ins[2][1],
                      op.ins[2][0],
                      op.ins[2][2:],
                      stride=op.attr['stride'],
                      padding=op.attr['padding'],
                      bias=(len(op.outs[-1]) == 1))
    layer.train()
    layer.cuda()
    for _ in range(50):
        y = layer(x)
        y.backward(torch.ones_like(y))
    return module_backward_helper(layer, dev, [x])


def profile_bn2d(op, dev):
    x = make_randn_tensor(op.ins[0], dev.type)
    x.requires_grad = True
    layer = nn.BatchNorm2d(op.ins[1][0])
    layer.train()
    return module_forward_helper(layer, dev, [x])


def profile_bn2d_bw(op, dev):
    x = make_randn_tensor(op.ins[1], dev.type)
    x.requires_grad = True
    layer = nn.BatchNorm2d(op.ins[2][0])
    layer.train()
    return module_backward_helper(layer, dev, [x])


def profile_layernorm(op, dev):
    x = make_randn_tensor(op.ins[0], dev.type)
    x.requires_grad = True
    layer = nn.LayerNorm(op.ins[1][0])
    layer.train()
    return module_forward_helper(layer, dev, [x])


def profile_layernorm_bw(op, dev):
    x = make_randn_tensor(op.ins[1], dev.type)
    x.requires_grad = True
    layer = nn.LayerNorm(op.ins[2][0])
    layer.train()
    return module_backward_helper(layer, dev, [x])


def profile_relu(op, dev):
    x = make_randn_tensor(op.ins[0], dev.type)
    x.requires_grad = True
    layer = nn.ReLU()
    layer.train()
    return module_forward_helper(layer, dev, [x])


def profile_relu_bw(op, dev):
    x = make_randn_tensor(op.ins[0], dev.type)
    x.requires_grad = True
    layer = nn.ReLU()
    layer.train()
    return module_backward_helper(layer, dev, [x])


# >>> activation functions
def openai_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))


def profile_activation(op, dev):
    if op.act == 'gelu':
        x = make_randn_tensor(op.ins[0], dev.type)
        x.requires_grad = True
        return function_forward_helper(openai_gelu, [x])
    elif op.act == 'sigmoid':
        x = make_randn_tensor(op.ins[0], dev.type)
        x.requires_grad = True
        layer = nn.Sigmoid()
        layer.train()
        return module_forward_helper(layer, dev, [x])
    return profile_relu(op, dev)


def profile_activation_bw(op, dev):
    if op.act == 'gelu':
        x = make_randn_tensor(op.ins[0], dev.type)
        x.requires_grad = True
        return function_backward_helper(openai_gelu, [x])
    elif op.act == 'sigmoid':
        x = make_randn_tensor(op.ins[0], dev.type)
        x.requires_grad = True
        layer = nn.Sigmoid()
        layer.train()
        return module_backward_helper(layer, dev, [x])
    return profile_relu_bw(op, dev)


def profile_dropout(op, dev):
    x = make_randn_tensor(op.ins[0], dev.type)
    x.requires_grad = True
    layer = nn.Dropout(p=op.attr['p'])
    layer.train()
    return module_forward_helper(layer, dev, [x])


def profile_dropout_bw(op, dev):
    x = make_randn_tensor(op.ins[0], dev.type)
    x.requires_grad = True
    layer = nn.Dropout(p=op.attr['p'])
    layer.train()
    return module_backward_helper(layer, dev, [x])


def profile_pool2d(op, dev):
    x = make_randn_tensor(op.ins[0], dev.type)
    x.requires_grad = True
    if op.attr['mode'] == 'max':
        layer = nn.MaxPool2d(op.attr['kernel_size'],
                             stride=op.attr['stride'],
                             padding=op.attr['padding'])
    else:
        layer = nn.AvgPool2d(op.attr['kernel_size'],
                             stride=op.attr['stride'],
                             padding=op.attr['padding'])
    layer.train()
    return module_forward_helper(layer, dev, [x])


def profile_pool2d_bw(op, dev):
    x = make_randn_tensor(op.outs[0], dev.type)
    x.requires_grad = True
    if op.attr['mode'] == 'max':
        layer = nn.MaxPool2d(op.attr['kernel_size'],
                             stride=op.attr['stride'],
                             padding=op.attr['padding'])
    else:
        layer = nn.AvgPool2d(op.attr['kernel_size'],
                             stride=op.attr['stride'],
                             padding=op.attr['padding'])
    layer.train()
    return module_backward_helper(layer, dev, [x])


def profile_adaptivepool2d(op, dev):
    x = make_randn_tensor(op.ins[0], dev.type)
    x.requires_grad = True
    if op.attr['mode'] == 'max':
        layer = nn.AdaptiveMaxPool2d(op.attr['output_size'])
    else:
        layer = nn.AdaptiveAvgPool2d(op.attr['output_size'])
    layer.train()
    return module_forward_helper(layer, dev, [x])


def profile_adaptivepool2d_bw(op, dev):
    x = make_randn_tensor(op.outs[0], dev.type)
    x.requires_grad = True
    if op.attr['mode'] == 'max':
        layer = nn.AdaptiveMaxPool2d(op.attr['output_size'])
    else:
        layer = nn.AdaptiveAvgPool2d(op.attr['output_size'])
    layer.train()
    return module_backward_helper(layer, dev, [x])


def profile_reshape(op, dev):
    x = make_randn_tensor(op.ins[0], dev.type, requires_grad=True)
    return tensor_forward_helper('reshape', dev, x, op.outs[0])


def profile_permute(op, dev):
    x = make_randn_tensor(op.ins[0], dev.type, requires_grad=True)
    return tensor_forward_helper('permute', dev, x, op.perm)


def profile_split(op, dev):
    x = make_randn_tensor(op.ins[0], dev.type, requires_grad=True)
    return tensor_forward_helper('split',
                                 dev,
                                 x,
                                 op.attr['size_or_sections'],
                                 dim=op.attr['dim'])


def profile_concat(op, dev):
    ins = []
    for shp in op.ins:
        ins.append(make_randn_tensor(shp, dev.type, requires_grad=True))
    return torch_forward_helper('cat', dev, [ins], dim=op.dim)

def profile_slice(op, dev):
    x = make_randn_tensor(op.ins[0], dev.type)
    x.requires_grad = True

    def fn_slice(x):
        return x[:, :op.nelements]
    return function_forward_helper(fn_slice, [x])


def profile_slice_bw(op, dev):
    x = make_randn_tensor(op.ins[0], dev.type)
    x.requires_grad = True

    def fn_slice(x):
        return x[:, :op.nelements]
    return function_backward_helper(fn_slice, [x])


def profile_softmax(op, dev):
    x = make_randn_tensor(op.ins[0], dev.type)
    x.requires_grad = True
    layer = nn.Softmax(dim=op.dim)
    layer.train()
    return module_forward_helper(layer, dev, [x])


def profile_softmax_bw(op, dev):
    x = make_randn_tensor(op.ins[0], dev.type)
    x.requires_grad = True
    layer = nn.Softmax(dim=op.dim)
    layer.train()
    return module_backward_helper(layer, dev, [x])


def profile_crossentropy(op, dev):
    x = make_randn_tensor(op.ins[0], dev.type)
    x.requires_grad = True
    target = make_randn_label(op.ins[1], dev.type, 1000)
    if op.loss_type == 'mse':
        target = target.float()
        layer = nn.MSELoss(reduction='mean')
    else:
        layer = nn.CrossEntropyLoss()
    layer.train()
    if x.ndim > 2:
        x = x.view(-1, x.size(-1))
    return module_forward_helper(layer, dev, [x, target.view(-1)])


def profile_crossentropy_bw(op, dev):
    x = make_randn_tensor(op.outs[0], dev.type)
    target = make_randn_label(op.ins[1], dev.type, 1000)
    x.requires_grad = True
    if op.loss_type == 'mse':
        target = target.float()
        layer = nn.MSELoss(reduction='mean')
    else:
        layer = nn.CrossEntropyLoss()
    layer.train()
    return module_backward_helper(
        layer, dev,
        [x.view(-1, x.size(-1)), target.view(-1)])


def attention_mask(x, mask):
    x = x / 8
    # x = torch.mul(x, mask) - 10000.0 * (1.0 - mask)
    x.masked_fill_(mask, -10000)
    return x


def profile_elementwise(op, dev):
    ins = []
    for shp in op.ins:
        ins.append(make_randn_tensor(shp, dev.type, requires_grad=True))
    if op.type == OpType.AttentionMask:
        ins[0].requires_grad = True
        ins[1] = make_randn_tensor([1, 1, shp[2], shp[3]],
                                   dev.type,
                                   requires_grad=False)
        ins[1] = ins[1] > 0.5
        return function_forward_helper(attention_mask, ins)
    return torch_forward_helper('add', dev, ins)


def profile_elementwise_bw(op, dev):
    ins = []
    if len(op.outs) == 2:
        for shp in op.outs:
            ins.append(make_randn_tensor(shp, dev.type, requires_grad=True))
    else:
        if len(op.ins) == 3:
            for shp in op.ins[1:]:
                ins.append(
                    make_randn_tensor(shp, dev.type, requires_grad=False))
        else:
            for shp in [op.outs[0], op.ins[0]]:
                ins.append(
                    make_randn_tensor(shp, dev.type, requires_grad=False))
        ins[0].requires_grad = True
    if op.type == OpType.AttentionMask:
        ins[0].requires_grad = True
        ins[1] = make_randn_tensor([1, 1, shp[2], shp[3]],
                                   dev.type,
                                   requires_grad=False)
        ins[1] = ins[1] > 0.5
        return function_backward_helper(attention_mask, ins)
    return torch_backward_helper('add', dev, ins)


def generate_dist_input_batch(
    ln_emb,
    n,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
    rand_data_dist='uniform',
    rand_data_min=0,
    rand_data_max=1,
    rand_data_mu=-1,
    rand_data_sigma=1,
):
    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for size in ln_emb:
        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        for _ in range(n):
            # num of sparse indices to be used per embedding (between
            if num_indices_per_lookup_fixed:
                sparse_group_size = np.int64(num_indices_per_lookup)
            else:
                # random between [1,num_indices_per_lookup])
                r = np.random.random(1)
                sparse_group_size = np.int64(
                    np.round(max([1.0], r * min(size, num_indices_per_lookup)))
                )
            # sparse indices to be used per embedding
            if rand_data_dist == "gaussian":
                if rand_data_mu == -1:
                    rand_data_mu = (rand_data_max + rand_data_min) / 2.0
                r = np.random.normal(rand_data_mu, rand_data_sigma, sparse_group_size)
                sparse_group = np.clip(r, rand_data_min, rand_data_max)
                sparse_group = np.unique(sparse_group).astype(np.int64)
            elif rand_data_dist == "uniform":
                r = np.random.random(sparse_group_size)
                sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int64))
            else:
                raise(rand_data_dist, "distribution is not supported. \
                     please select uniform or gaussian")

            # reset sparse_group_size in case some index duplicates were removed
            sparse_group_size = np.int64(sparse_group.size)
            # store lengths and indices
            lS_batch_offsets += [offset]
            lS_batch_indices += sparse_group.tolist()
            # update offset for next iteration
            offset += sparse_group_size
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets))
        lS_emb_indices.append(torch.tensor(lS_batch_indices))

    return (lS_emb_offsets, lS_emb_indices)


def profile_embedding(op, dev):
    if op.attr is None:
        x = make_randn_label(op.ins[0], dev.type, op.ins[1][0])
        layer = nn.Embedding(*op.ins[1])
        layer.train()
        return module_forward_helper(layer, dev, [x])
    else:
        x = generate_dist_input_batch(
            ln_emb=[op.ins[1][0]] * 20,
            n=op.ins[0][0],
            num_indices_per_lookup=op.attr['num_indices_per_lookup'],
            num_indices_per_lookup_fixed=True,
            rand_data_dist='uniform'
        )
        layer = nn.EmbeddingBag(*op.ins[1], mode=op.attr['mode'])
        layer.train()
        x = [(x2_, x1_) for (x1_, x2_) in zip(x[0], x[1])]
        return module_forward_helper(layer, dev, x, gen=False)


def profile_embedding_bw(op, dev):
    if op.attr is None:
        x = make_randn_label(op.ins[1], dev.type, op.outs[0][0])
        layer = nn.Embedding(*op.outs[0])
        layer.train()
        return module_backward_helper(layer, dev, [x])
    else:
        x = generate_dist_input_batch(
            ln_emb=[op.outs[0][0]] * 20,
            n=op.ins[0][0],
            num_indices_per_lookup=op.attr['num_indices_per_lookup'],
            num_indices_per_lookup_fixed=True,
            rand_data_dist='uniform'
        )
        layer = nn.EmbeddingBag(*op.outs[0], mode=op.attr['mode'])
        layer.train()
        x = [(x2_, x1_) for (x1_, x2_) in zip(x[0], x[1])]
        return module_backward_helper(layer, dev, x, gen=False)

def profile_sgd(op, dev):
    param = make_randn_tensor(op.ins[0], dev.type)
    param.grad = make_randn_tensor(op.ins[0], dev.type)
    opt = torch.optim.SGD([param],
                          lr=op.attr['lr'],
                          momentum=op.attr['momentum'])
    return optimizer_step(opt)


def profile_adam(op, dev):
    param = make_randn_tensor(op.ins[0], dev.type)
    param.grad = make_randn_tensor(op.ins[0], dev.type)
    opt = torch.optim.Adam([param])
    return optimizer_step(opt)
