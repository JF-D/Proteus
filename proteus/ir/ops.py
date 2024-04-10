from proteus import OpType
from proteus.simulator.cost_model import register_op_cost_model
from proteus.simulator.roofline_model import *
from proteus.simulator.profile_model import *
from .node import Op
from .op_helper import IterSpace


class Linear(Op):
    def __init__(self, ins, outs, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.Linear
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_linear(self)
        return self.iter_space


class LinearBW(Op):
    def __init__(self, ins, outs, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.LinearBW
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_linear_bw(self)
        return self.iter_space


class Matmul(Op):
    def __init__(self, ins, outs, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.Matmul
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_matmul(self)
        return self.iter_space


class MatmulBW(Op):
    def __init__(self, ins, outs, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.MatmulBW
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_matmul_bw(self)
        return self.iter_space


class Conv2d(Op):
    def __init__(self, ins, outs, attr=None, name=''):
        super().__init__(ins, outs, attr=attr, name=name)
        self.type = OpType.Conv2d
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_conv2d(self)
        return self.iter_space


class Conv2dBW(Op):
    def __init__(self, ins, outs, attr=None, name=''):
        super().__init__(ins, outs, attr=attr, name=name)
        self.type = OpType.Conv2dBW
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_conv2d_bw(self)
        return self.iter_space


class BatchNorm2d(Op):
    def __init__(self, ins, outs, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.BatchNorm2d
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_batchnorm2d(self)
        return self.iter_space


class BatchNorm2dBW(Op):
    def __init__(self, ins, outs, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.BatchNorm2dBW
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_batchnorm2d_bw(self)
        return self.iter_space


class LayerNorm(Op):
    def __init__(self, ins, outs, normalized_shape, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.LayerNorm
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])
        self.normalized_shape = normalized_shape if isinstance(
            normalized_shape, (tuple, list)) else [normalized_shape]

    def get_iter_space(self):
        self.iter_space = IterSpace.get_layernorm(self)
        return self.iter_space


class LayerNormBW(Op):
    def __init__(self, ins, outs, normalized_shape, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.LayerNormBW
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])
        self.normalized_shape = normalized_shape if isinstance(
            normalized_shape, (tuple, list)) else [normalized_shape]

    def get_iter_space(self):
        self.iter_space = IterSpace.get_layernorm_bw(self)
        return self.iter_space


class ReLU(Op):
    def __init__(self, ins, outs, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.ReLU
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_elementwise(self)
        return self.iter_space


class ReLUBW(Op):
    def __init__(self, ins, outs, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.ReLUBW
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_elementwise(self)
        return self.iter_space


class Activation(Op):
    def __init__(self, ins, outs, act, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.Activation
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])
        self.act = act

    def get_iter_space(self):
        self.iter_space = IterSpace.get_elementwise(self)
        return self.iter_space


class ActivationBW(Op):
    def __init__(self, ins, outs, act, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.ActivationBW
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])
        self.act = act

    def get_iter_space(self):
        self.iter_space = IterSpace.get_elementwise(self)
        return self.iter_space


class Dropout(Op):
    def __init__(self, ins, outs, attr=None, name=''):
        super().__init__(ins, outs, attr=attr, name=name)
        self.type = OpType.Dropout
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_elementwise(self)
        return self.iter_space


class DropoutBW(Op):
    def __init__(self, ins, outs, attr=None, name=''):
        super().__init__(ins, outs, attr=attr, name=name)
        self.type = OpType.DropoutBW
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_elementwise(self)
        return self.iter_space


class Pool2d(Op):
    def __init__(self, ins, outs, attr=None, name=''):
        super().__init__(ins, outs, attr=attr, name=name)
        self.type = OpType.Pool2d
        self.mode = attr['mode']
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_pool2d(self)
        return self.iter_space


class Pool2dBW(Op):
    def __init__(self, ins, outs, attr=None, name=''):
        super().__init__(ins, outs, attr=attr, name=name)
        self.type = OpType.Pool2dBW
        self.mode = attr['mode']
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_pool2d(self)
        return self.iter_space


class AdaptivePool2d(Op):
    def __init__(self, ins, outs, attr=None, name=''):
        super().__init__(ins, outs, attr=attr, name=name)
        self.type = OpType.AdaptivePool2d
        self.mode = attr['mode']
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_pool2d(self)
        return self.iter_space


class AdaptivePool2dBW(Op):
    def __init__(self, ins, outs, attr=None, name=''):
        super().__init__(ins, outs, attr=attr, name=name)
        self.type = OpType.AdaptivePool2dBW
        self.mode = attr['mode']
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_pool2d(self)
        return self.iter_space


class Reshape(Op):
    def __init__(self, ins, outs, bw=False, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.Reshape
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])
        self.bw = bw

    def get_iter_space(self):
        if self.bw:
            self.iter_space = IterSpace.get_reshape_bw(self)
        else:
            self.iter_space = IterSpace.get_reshape(self)
        return self.iter_space


class Permute(Op):
    def __init__(self, ins, outs, perm, bw=False, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.Permute
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])
        self.bw = bw
        if self.bw:
            self.perm = [0] * len(perm)
            for i, p in enumerate(perm):
                self.perm[p] = i
        else:
            self.perm = perm

    def get_iter_space(self):
        if self.bw:
            self.iter_space = IterSpace.get_permute_bw(self)
        else:
            self.iter_space = IterSpace.get_permute(self)
        return self.iter_space


class Split(Op):
    def __init__(self, ins, outs, dim, attr=None, name=''):
        super().__init__(ins, outs, attr=attr, name=name)
        self.type = OpType.Split
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])
        self.dim = dim

    def get_iter_space(self):
        self.iter_space = IterSpace.get_split(self)
        return self.iter_space


class Concat(Op):
    def __init__(self, ins, outs, dim, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.Concat
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])
        self.dim = dim

    def get_iter_space(self):
        self.iter_space = IterSpace.get_concat(self)
        return self.iter_space


class SliceFW(Op):
    def __init__(self, ins, outs, nelements, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.SliceFW
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

        self.nelements = nelements

    def get_iter_space(self):
        self.iter_space = IterSpace.get_slice(self)
        return self.iter_space


class SliceBW(Op):
    def __init__(self, ins, outs, nelements, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.SliceBW
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

        self.nelements = nelements

    def get_iter_space(self):
        self.iter_space = IterSpace.get_slice(self)
        return self.iter_space


class Softmax(Op):
    def __init__(self, ins, outs, dim=0, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.Softmax
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])
        self.dim = dim

    def get_iter_space(self):
        self.iter_space = IterSpace.get_softmax(self)
        return self.iter_space


class SoftmaxBW(Op):
    def __init__(self, ins, outs, dim=0, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.SoftmaxBW
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])
        self.dim = dim

    def get_iter_space(self):
        self.iter_space = IterSpace.get_softmax(self)
        return self.iter_space


class CrossEntropy(Op):
    def __init__(self, ins, outs, loss_type=None, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.CrossEntropy
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

        self.loss_type = loss_type

    def get_iter_space(self):
        self.iter_space = IterSpace.get_crossentropy(self)
        return self.iter_space


class CrossEntropyBW(Op):
    def __init__(self, ins, outs, loss_type=None, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.CrossEntropyBW
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

        self.loss_type = loss_type

    def get_iter_space(self):
        self.iter_space = IterSpace.get_crossentropy_bw(self)
        return self.iter_space


class Elementwise(Op):
    def __init__(self, ins, outs, type: str, name=''):
        super().__init__(ins, outs, name=name)
        if type.lower() == 'add':
            self.type = OpType.Add
        elif type.lower() == 'sub':
            self.type = OpType.Sub
        elif type.lower() == 'mul':
            self.type = OpType.Mul
        elif type.lower() == 'div':
            self.type = OpType.Div
        elif type.lower() == 'sqrt':
            self.type = OpType.Sqrt
        elif type.lower() == 'attention_mask':
            self.type = OpType.AttentionMask
        else:
            assert False, 'Unknown elementwise type: {}'.format(type)
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_binary_elementwise(self)
        return self.iter_space


class ElementwiseBW(Op):
    def __init__(self, ins, outs, type: str, input_shape, name=''):
        super().__init__(ins, outs, name=name)
        if type.lower() == 'add':
            self.type = OpType.AddBW
        elif type.lower() == 'sub':
            self.type = OpType.SubBW
        elif type.lower() == 'mul':
            self.type = OpType.MulBW
        elif type.lower() == 'div':
            self.type = OpType.DivBW
        elif type.lower() == 'sqrt':
            self.type = OpType.SqrtBW
        elif type.lower() == 'attention_mask':
            self.type = OpType.AttentionMaskBW
        else:
            assert False, 'Unknown elementwise type: {}'.format(type)
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

        self.saved_input_shape = input_shape

    def get_iter_space(self):
        self.iter_space = IterSpace.get_binary_elementwise_bw(self)
        return self.iter_space


class Embedding(Op):
    def __init__(self, ins, outs, attr=None, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.Embedding
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

        self.attr = attr

    def get_iter_space(self):
        self.iter_space = IterSpace.get_embedding(self)
        return self.iter_space


class EmbeddingBW(Op):
    def __init__(self, ins, outs, attr=None, name=''):
        super().__init__(ins, outs, name=name)
        self.type = OpType.EmbeddingBW
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

        self.attr = attr

    def get_iter_space(self):
        self.iter_space = IterSpace.get_embedding_bw(self)
        return self.iter_space


class SGD(Op):
    def __init__(self, ins, outs, attr=None, name=''):
        super().__init__(ins, outs, attr=attr, name=name)
        self.type = OpType.SGDApply
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_elementwise(self)
        return self.iter_space


class Adam(Op):
    def __init__(self, ins, outs, attr=None, name=''):
        super().__init__(ins, outs, attr=attr, name=name)
        self.type = OpType.AdamApply
        self.ins = tuple([x.size() for x in ins])
        self.outs = tuple([y.size() for y in outs])

    def get_iter_space(self):
        self.iter_space = IterSpace.get_elementwise(self)
        return self.iter_space


register_op_cost_model(Linear, 'roofline', roofline_linear)
register_op_cost_model(LinearBW, 'roofline', roofline_linear_bw)
register_op_cost_model(Matmul, 'roofline', roofline_matmul)
register_op_cost_model(MatmulBW, 'roofline', roofline_matmul_bw)
register_op_cost_model(Conv2d, 'roofline', roofline_conv2d)
register_op_cost_model(Conv2dBW, 'roofline', roofline_conv2d_bw)
register_op_cost_model(BatchNorm2d, 'roofline', roofline_mem_bound)
register_op_cost_model(BatchNorm2dBW, 'roofline', roofline_mem_bound)
register_op_cost_model(LayerNorm, 'roofline', roofline_mem_bound)
register_op_cost_model(LayerNormBW, 'roofline', roofline_mem_bound)
register_op_cost_model(ReLU, 'roofline', roofline_mem_bound)
register_op_cost_model(ReLUBW, 'roofline', roofline_mem_bound)
register_op_cost_model(Activation, 'roofline', roofline_mem_bound)
register_op_cost_model(ActivationBW, 'roofline', roofline_mem_bound)
register_op_cost_model(Dropout, 'roofline', roofline_mem_bound)
register_op_cost_model(DropoutBW, 'roofline', roofline_mem_bound)
register_op_cost_model(Pool2d, 'roofline', roofline_mem_bound)
register_op_cost_model(Pool2dBW, 'roofline', roofline_mem_bound)
register_op_cost_model(AdaptivePool2d, 'roofline', roofline_mem_bound)
register_op_cost_model(AdaptivePool2dBW, 'roofline', roofline_mem_bound)
register_op_cost_model(Reshape, 'roofline', roofline_mem_bound)
register_op_cost_model(Permute, 'roofline', roofline_mem_bound)
register_op_cost_model(Split, 'roofline', roofline_mem_bound)
register_op_cost_model(Concat, 'roofline', roofline_mem_bound)
register_op_cost_model(Softmax, 'roofline', roofline_mem_bound)
register_op_cost_model(SoftmaxBW, 'roofline', roofline_mem_bound)
register_op_cost_model(CrossEntropy, 'roofline', roofline_mem_bound)
register_op_cost_model(CrossEntropyBW, 'roofline', roofline_mem_bound)
register_op_cost_model(Elementwise, 'roofline', roofline_mem_bound)
register_op_cost_model(ElementwiseBW, 'roofline', roofline_mem_bound)
register_op_cost_model(Embedding, 'roofline', roofline_embedding)
register_op_cost_model(EmbeddingBW, 'roofline', roofline_embedding_bw)
register_op_cost_model(SGD, 'roofline', roofline_mem_bound)
register_op_cost_model(Adam, 'roofline', roofline_mem_bound)

register_op_cost_model(Linear, 'profile', profile_linear)
register_op_cost_model(LinearBW, 'profile', profile_linear_bw)
register_op_cost_model(Matmul, 'profile', profile_matmul)
register_op_cost_model(MatmulBW, 'profile', profile_matmul_bw)
register_op_cost_model(Conv2d, 'profile', profile_conv2d)
register_op_cost_model(Conv2dBW, 'profile', profile_conv2d_bw)
register_op_cost_model(BatchNorm2d, 'profile', profile_bn2d)
register_op_cost_model(BatchNorm2dBW, 'profile', profile_bn2d_bw)
register_op_cost_model(LayerNorm, 'profile', profile_layernorm)
register_op_cost_model(LayerNormBW, 'profile', profile_layernorm_bw)
register_op_cost_model(ReLU, 'profile', profile_relu)
register_op_cost_model(ReLUBW, 'profile', profile_relu_bw)
register_op_cost_model(Activation, 'profile', profile_activation, cache_key_fn=lambda op: '{}_{}_{}'.format(op.act, op.ins, op.outs))
register_op_cost_model(ActivationBW, 'profile', profile_activation_bw, cache_key_fn=lambda op: '{}_{}_{}'.format(op.act, op.ins, op.outs))
register_op_cost_model(Dropout, 'profile', profile_dropout)
register_op_cost_model(DropoutBW, 'profile', profile_dropout_bw)
register_op_cost_model(Pool2d, 'profile', profile_pool2d)
register_op_cost_model(Pool2dBW, 'profile', profile_pool2d_bw)
register_op_cost_model(AdaptivePool2d, 'profile', profile_adaptivepool2d)
register_op_cost_model(AdaptivePool2dBW, 'profile', profile_adaptivepool2d_bw)
register_op_cost_model(Reshape, 'profile', profile_reshape)
register_op_cost_model(Permute, 'profile', profile_permute)
register_op_cost_model(Split, 'profile', profile_split)
register_op_cost_model(Concat, 'profile', profile_concat)
register_op_cost_model(SliceFW, 'profile', profile_slice)
register_op_cost_model(SliceBW, 'profile', profile_slice_bw)
register_op_cost_model(Softmax, 'profile', profile_softmax)
register_op_cost_model(SoftmaxBW, 'profile', profile_softmax_bw)
register_op_cost_model(CrossEntropy, 'profile', profile_crossentropy)
register_op_cost_model(CrossEntropyBW, 'profile', profile_crossentropy_bw)
register_op_cost_model(Elementwise, 'profile', profile_elementwise)
register_op_cost_model(ElementwiseBW, 'profile', profile_elementwise_bw)
register_op_cost_model(Embedding, 'profile', profile_embedding)
register_op_cost_model(EmbeddingBW, 'profile', profile_embedding_bw)
register_op_cost_model(SGD, 'profile', profile_sgd)
register_op_cost_model(Adam, 'profile', profile_adam)
