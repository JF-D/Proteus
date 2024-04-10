import math
from proteus import IterType


class IterSpace(object):

    def __init__(self, op, iters, in_iters, out_iters):
        super().__init__()
        self.type = op.type
        self.iters = iters
        self.in_iters = in_iters
        self.out_iters = out_iters

        bounds = dict.fromkeys(range(len(self.iters)), [])
        if IterType.CONTINUOUS in self.iters:
            # this is reshape op
            assert len(in_iters) == 1 and len(out_iters) == 1
            if len(op.ins[0]) == len(self.iters):
                bounds = list(op.ins[0])
            else:
                bounds = list(op.outs[0])
        else:
            bounds = [math.inf] * len(self.iters)
            for x_it, x_in in zip(self.in_iters, op.ins):
                for x_it_id, x_in_b in zip(x_it, x_in):
                    bounds[x_it_id] = min(bounds[x_it_id], x_in_b)
            for y_it, y_out in zip(self.out_iters, op.outs):
                for y_it_id, y_out_b in zip(y_it, y_out):
                    bounds[y_it_id] = min(bounds[y_it_id], y_out_b)
        self.bounds = tuple(bounds)

    def __len__(self):
        return len(self.iters)

    def __repr__(self):
        string = f'{self.type}: {self.iters}\n'
        string += f'    {self.in_iters}\n'
        string += f'    {self.out_iters}\n'
        return string

    @staticmethod
    def get_linear(op):
        ins, outs = (op.ins, op.outs)
        iters = [IterType.BATCH
                 ] + [IterType.OTHER] * (len(ins[0]) - 1) + [IterType.REDUCE]
        in_x = tuple(list(range(len(ins[0]) - 1)) + [len(ins[0])])
        in_w = (len(iters) - 2, len(iters) - 1)
        in_iters = (in_x, in_w)
        if len(ins) > 2:
            in_b = (len(iters) - 2, )
            in_iters = (in_x, in_w, in_b)
        out_y = tuple(range(len(outs[0])))
        out_iters = (out_y, )
        return IterSpace(op, iters, in_iters, out_iters)

    @staticmethod
    def get_linear_bw(op):
        ins, outs = (op.ins, op.outs)
        iters = [IterType.BATCH
                 ] + [IterType.OTHER] * (len(ins[1]) - 1) + [IterType.REDUCE]
        in_dy = tuple(range(len(ins[0])))
        in_x = tuple(list(range(len(ins[1]) - 1)) + [len(ins[1])])
        in_w = (len(iters) - 2, len(iters) - 1)
        in_iters = (in_dy, in_x, in_w)
        if ins[1] == outs[0]:
            out_iters = [in_x, in_w]
        else:
            out_iters = [in_w]
        if len(outs[-1]) == 1:
            out_db = (len(iters) - 2, )
            out_iters.append(out_db)
        return IterSpace(op, iters, in_iters, tuple(out_iters))

    @staticmethod
    def get_matmul(op):
        ins, outs = (op.ins, op.outs)
        iters = [IterType.BATCH] + [IterType.OTHER] * len(ins[0])
        assert len(iters) >= 3
        iters[-2] = IterType.REDUCE
        in_x1 = tuple(range(len(ins[0])))
        x_indices = list(range(len(ins[0])))
        in_x2 = tuple(x_indices[:len(ins[1]) - 2] +
                      [len(iters) - 2, len(iters) - 1])
        in_iters = (in_x1, in_x2)
        out_y = tuple(x_indices[:-1] + [len(iters) - 1])
        out_iters = (out_y, )
        return IterSpace(op, iters, in_iters, out_iters)

    @staticmethod
    def get_matmul_bw(op):
        ins, outs = (op.ins, op.outs)
        iters = [IterType.BATCH] + [IterType.OTHER] * len(ins[1])
        assert len(iters) >= 3
        iters[-2] = IterType.REDUCE
        in_x1 = tuple(range(len(ins[1])))
        x_indices = list(range(len(ins[1])))
        in_x2 = tuple(x_indices[:len(ins[2]) - 2] +
                      [len(iters) - 2, len(iters) - 1])
        in_dy = tuple(x_indices[:-1] + [len(iters) - 1])
        in_iters = (in_dy, in_x1, in_x2)
        out_iters = (in_x1, in_x2)
        return IterSpace(op, iters, in_iters, out_iters)

    @staticmethod
    def get_conv2d(op):
        # (N, C_out, C_in, H, W, K1, K2)
        ndims = len(op.ins[0]) - 2
        pd = [IterType.BATCH, IterType.OTHER, IterType.REDUCE
              ] + [IterType.OPAQUE] * ndims + [IterType.OPAQUE] * ndims
        in_x = tuple([0, 2] + list(range(3, 3 + ndims)))
        in_w = tuple([1, 2] + list(range(len(pd) - ndims, len(pd))))
        in_indices = (in_x, in_w)
        if len(op.ins) > 2:
            in_b = (1, )
            in_indices = (in_x, in_w, in_b)
        out_y = tuple([0, 1] + list(range(3, 3 + ndims)))
        out_indices = (out_y, )
        return IterSpace(op, pd, in_indices, out_indices)

    @staticmethod
    def get_conv2d_bw(op):
        # (N, C_out, C_in, H, W, K1, K2)
        ndims = len(op.ins[0]) - 2
        pd = [IterType.BATCH, IterType.OTHER, IterType.REDUCE
              ] + [IterType.OPAQUE] * ndims + [IterType.OPAQUE] * ndims
        in_dy = tuple([0, 1] + list(range(3, 3 + ndims)))
        in_x = tuple([0, 2] + list(range(3, 3 + ndims)))
        in_w = tuple([1, 2] + list(range(len(pd) - ndims, len(pd))))
        in_indices = (in_dy, in_x, in_w)
        if op.ins[1] == op.outs[0]:
            out_indices = [in_x, in_w]
        else:
            out_indices = [in_w]
        if len(op.outs[-1]) == 1:
            out_db = (1, )
            out_indices.append(out_db)
        return IterSpace(op, pd, in_indices, tuple(out_indices))

    @staticmethod
    def get_batchnorm2d(op):
        pd = [IterType.BATCH, IterType.OTHER
              ] + [IterType.REDUCE] * (len(op.ins[0]) - 2)
        in_x = tuple(range(len(op.ins[0])))
        in_indices = (in_x, )
        if len(op.ins) > 1:
            in_indices = tuple([in_x] + [(1, )
                                         for _ in range(len(op.ins) - 1)])
        out_y = tuple(range(len(op.outs[0])))
        out_indices = (out_y, )
        if len(op.outs) > 1:
            out_indices = tuple([out_y] + [(1, )
                                           for _ in range(len(op.outs) - 1)])
        return IterSpace(op, pd, in_indices, out_indices)

    @staticmethod
    def get_batchnorm2d_bw(op):
        pd = [IterType.BATCH, IterType.OTHER
              ] + [IterType.REDUCE] * (len(op.ins[0]) - 2)
        in_dy = tuple(range(len(op.ins[0])))
        in_x = tuple(range(len(op.ins[1])))
        in_indices = (in_dy, in_x)
        if len(op.ins) > 1:
            in_indices = tuple([in_dy, in_x, (1, )])
        out_indices = (in_x, )
        if len(op.outs) > 1:
            out_indices = (in_x, (1, ), (1, ))
        return IterSpace(op, pd, in_indices, out_indices)

    @staticmethod
    def get_layernorm(op):
        iters = [IterType.BATCH] + [IterType.OTHER] * (
            len(op.ins[0]) - len(op.normalized_shape) -
            1) + [IterType.OPAQUE] * len(op.normalized_shape)
        in_x = tuple(range(len(op.ins[0])))
        normalize_iters = range(
            len(op.ins[0]) - len(op.normalized_shape), len(iters))
        in_iters = (in_x, ) + tuple(
            [tuple(normalize_iters) for _ in op.ins[1:]])
        out_iters = (tuple(range(len(op.outs[0]))), )
        return IterSpace(op, iters, in_iters, out_iters)

    @staticmethod
    def get_layernorm_bw(op):
        iters = [IterType.BATCH] + [IterType.OTHER] * (
            len(op.ins[0]) - len(op.normalized_shape) -
            1) + [IterType.OPAQUE] * len(op.normalized_shape)
        in_x = tuple(range(len(op.ins[0])))
        normalize_iters = range(
            len(op.ins[0]) - len(op.normalized_shape), len(iters))
        in_iters = (in_x, in_x) + tuple(
            [tuple(normalize_iters) for _ in op.ins[2:]])
        out_iters = (in_x, ) + tuple(
            [tuple(normalize_iters) for _ in op.outs[1:]])
        return IterSpace(op, iters, in_iters, out_iters)

    @staticmethod
    def get_pool2d(op):
        iters = [IterType.BATCH, IterType.OTHER
                 ] + [IterType.OPAQUE] * (len(op.ins[0]) - 2)
        in_iters = (tuple(range(len(op.ins[0]))), )
        out_iters = (tuple(range(len(op.outs[0]))), )
        return IterSpace(op, iters, in_iters, out_iters)

    @staticmethod
    def get_reshape(op):
        old_shape, new_shape = op.ins[0], op.outs[0]
        iters, in_iter, out_iter = [], [], []

        st = 0
        for i in range(max(len(old_shape), len(new_shape))):
            if i >= len(old_shape) or i >= len(new_shape):
                iters.append(IterType.OPAQUE)
                if i < len(old_shape):
                    in_iter.append(i)
                if i < len(new_shape):
                    out_iter.append(i)
                continue
            if old_shape[i] == new_shape[i]:
                iters.append(IterType.BATCH if i == 0 else IterType.OTHER)
                in_iter.append(i)
                out_iter.append(i)
            else:
                iters.append(IterType.CONTINUOUS)
                st = i
                break
        iters_ed, in_iter_ed, out_iter_ed = [], [], []
        for i in range(-1, st - max(len(old_shape), len(new_shape)), -1):
            if old_shape[i] == new_shape[i]:
                iters_ed.insert(0, IterType.OTHER)
                in_iter_ed.insert(0, i)
                out_iter_ed.insert(0, i)
            else:
                break
        iters = iters + iters_ed
        in_iter_ed = [len(iters) + k for k in in_iter_ed]
        out_iter_ed = [len(iters) + k for k in out_iter_ed]
        in_iter = in_iter + [st] * (len(old_shape) - len(in_iter) -
                                    len(in_iter_ed)) + in_iter_ed
        out_iter = out_iter + [st] * (len(new_shape) - len(out_iter) -
                                      len(out_iter_ed)) + out_iter_ed

        assert len(in_iter) != len(out_iter)
        new_iters, iter_index_map = [], {}
        max_len = max(len(in_iter), len(out_iter))
        for i in range(len(iters)):
            new_iters.append(iters[i])
            iter_index_map[i] = len(new_iters) - 1
            if iters[i] == IterType.CONTINUOUS:
                new_iters = new_iters + [IterType.CONTINUOUS] * (max_len - len(iters))
        if len(in_iter) == max_len:
            new_in_iter = list(range(len(in_iter)))
        else:
            new_in_iter = [iter_index_map[i] for i in in_iter]
        if len(out_iter) == max_len:
            new_out_iter = list(range(len(out_iter)))
        else:
            new_out_iter = [iter_index_map[i] for i in out_iter]
        # return IterSpace(op, iters, (tuple(in_iter), ), (tuple(out_iter), ))
        return IterSpace(op, new_iters, (tuple(new_in_iter), ), (tuple(new_out_iter), ))

    @staticmethod
    def get_reshape_bw(op):
        old_shape, new_shape = op.outs[0], op.ins[0]
        iters, in_iter, out_iter = [], [], []

        st = 0
        for i in range(max(len(old_shape), len(new_shape))):
            if i >= len(old_shape) or i >= len(new_shape):
                iters.append(IterType.OPAQUE)
                if i < len(old_shape):
                    in_iter.append(i)
                if i < len(new_shape):
                    out_iter.append(i)
                continue
            if old_shape[i] == new_shape[i]:
                iters.append(IterType.BATCH if i == 0 else IterType.OTHER)
                in_iter.append(i)
                out_iter.append(i)
            else:
                iters.append(IterType.CONTINUOUS)
                st = i
                break
        iters_ed, in_iter_ed, out_iter_ed = [], [], []
        for i in range(-1, st - max(len(old_shape), len(new_shape)), -1):
            if old_shape[i] == new_shape[i]:
                iters_ed.insert(0, IterType.OTHER)
                in_iter_ed.insert(0, i)
                out_iter_ed.insert(0, i)
            else:
                break
        iters = iters + iters_ed
        in_iter_ed = [len(iters) + k for k in in_iter_ed]
        out_iter_ed = [len(iters) + k for k in out_iter_ed]
        in_iter = in_iter + [st] * (len(old_shape) - len(in_iter) -
                                    len(in_iter_ed)) + in_iter_ed
        out_iter = out_iter + [st] * (len(new_shape) - len(out_iter) -
                                      len(out_iter_ed)) + out_iter_ed
        assert len(in_iter) != len(out_iter)
        new_iters, iter_index_map = [], {}
        max_len = max(len(in_iter), len(out_iter))
        for i in range(len(iters)):
            new_iters.append(iters[i])
            iter_index_map[i] = len(new_iters) - 1
            if iters[i] == IterType.CONTINUOUS:
                new_iters = new_iters + [IterType.CONTINUOUS] * (max_len - len(iters))
        if len(in_iter) == max_len:
            new_in_iter = list(range(len(in_iter)))
        else:
            new_in_iter = [iter_index_map[i] for i in in_iter]
        if len(out_iter) == max_len:
            new_out_iter = list(range(len(out_iter)))
        else:
            new_out_iter = [iter_index_map[i] for i in out_iter]
        # return IterSpace(op, iters, (tuple(out_iter), ), (tuple(in_iter), ))
        return IterSpace(op, new_iters, (tuple(new_out_iter), ), (tuple(new_in_iter), ))

    @staticmethod
    def get_permute(op):
        iters = [IterType.BATCH] + [IterType.OTHER] * (len(op.ins[0]) - 1)
        in_iters = (tuple(range(len(op.ins[0]))), )
        out_iters = (tuple(op.perm), )
        return IterSpace(op, iters, in_iters, out_iters)

    @staticmethod
    def get_permute_bw(op):
        iters = [IterType.BATCH] + [IterType.OTHER] * (len(op.ins[0]) - 1)
        out_iters = (tuple(range(len(op.outs[0]))), )
        perm = [0] * len(op.perm)
        for i, p in enumerate(op.perm):
            perm[p] = i
        in_iters = (tuple(perm), )
        return IterSpace(op, iters, in_iters, out_iters)

    @staticmethod
    def get_split(op):
        iters = [IterType.BATCH] + [IterType.OTHER] * (len(op.ins[0]) - 1)
        iters[op.dim] = IterType.OPAQUE
        in_iters = (tuple(range(len(op.ins[0]))), )
        out_iters = []
        indices = list(range(len(op.ins[0])))
        for i in range(len(op.outs)):
            iters.append(IterType.OPAQUE)
            indices[op.dim] = len(iters) - 1
            out_iters.append(tuple(indices))
        out_iters = tuple(out_iters)
        return IterSpace(op, iters, in_iters, out_iters)

    @staticmethod
    def get_concat(op):
        iters = [IterType.BATCH] + [IterType.OTHER] * (len(op.outs[0]) - 1)
        iters[op.dim] = IterType.OPAQUE
        out_iters = (tuple(range(len(op.outs[0]))), )
        in_iters = []
        indices = list(range(len(op.outs[0])))
        for i in range(len(op.ins)):
            iters.append(IterType.OPAQUE)
            indices[op.dim] = len(iters) - 1
            in_iters.append(tuple(indices))
        in_iters = tuple(in_iters)
        return IterSpace(op, iters, in_iters, out_iters)

    @staticmethod
    def get_slice(op):
        iters = [IterType.BATCH] + [IterType.OPAQUE] * (len(op.ins[0]) - 1)
        in_iters = tuple([tuple(range(len(op.ins[0]))) for _ in op.ins])
        out_iters = (tuple(range(len(op.outs[0]))), )
        return IterSpace(op, iters, in_iters, out_iters)

    @staticmethod
    def get_softmax(op):
        iters = [IterType.BATCH] + [IterType.OPAQUE] * (len(op.ins[0]) - 1)
        iters[op.dim] = [IterType.OTHER]
        in_iters = tuple([tuple(range(len(op.ins[0]))) for _ in op.ins])
        out_iters = (tuple(range(len(op.outs[0]))), )
        return IterSpace(op, iters, in_iters, out_iters)

    @staticmethod
    def get_crossentropy(op):
        iters = [IterType.BATCH] + [IterType.OPAQUE] * (len(op.ins[0]) - 1)
        in_iters = (tuple(range(len(op.ins[0]))), tuple(range(len(op.ins[1]))))
        out_iters = (tuple(range(len(op.outs[0]))), )
        return IterSpace(op, iters, in_iters, out_iters)

    @staticmethod
    def get_crossentropy_bw(op):
        iters = [IterType.BATCH] + [IterType.OPAQUE] * (len(op.outs[0]) - 1)
        in_iters = (tuple(range(len(op.ins[0]))), tuple(range(len(op.ins[1]))))
        out_iters = (tuple(range(len(op.outs[0]))), )
        return IterSpace(op, iters, in_iters, out_iters)

    @staticmethod
    def get_elementwise(op):
        ins, outs = (op.ins, op.outs)
        iters = [IterType.BATCH] + [IterType.OTHER] * (len(ins[0]) - 1)
        in_iters = tuple([tuple(range(len(x))) for x in ins])
        out_iters = tuple([tuple(range(len(y))) for y in outs])
        return IterSpace(op, iters, in_iters, out_iters)

    @staticmethod
    def get_binary_elementwise(op):
        ins, outs = (op.ins, op.outs)
        iters = [IterType.BATCH] + [IterType.OTHER] * (len(outs[0]) - 1)
        for idx, dim_len in enumerate(outs[0]):
            if dim_len == 1:
                iters[idx] = IterType.OPAQUE
        expanded_ins = [1] * (len(iters) - len(ins[0])) + list(ins[0])
        for idx, dim_len in enumerate(expanded_ins):
            if dim_len == 1:
                iters[idx] = IterType.OPAQUE
        expanded_ins = [1] * (len(iters) - len(ins[1])) + list(ins[1])
        for idx, dim_len in enumerate(expanded_ins):
            if dim_len == 1:
                iters[idx] = IterType.OPAQUE
        in_iters = tuple(
            [tuple(range(len(iters) - len(x), len(iters))) for x in ins])
        out_iters = tuple([tuple(range(len(y))) for y in outs])
        return IterSpace(op, iters, in_iters, out_iters)

    @staticmethod
    def get_binary_elementwise_bw(op):
        ins, outs = (op.ins, op.outs)
        iters = [IterType.BATCH] + [IterType.OTHER] * (len(ins[0]) - 1)
        for idx, dim_len in enumerate(ins[0]):
            if dim_len == 1:
                iters[idx] = IterType.OPAQUE
        expanded_outs = [1] * (len(iters) - len(outs[0])) + list(outs[0])
        for idx, dim_len in enumerate(expanded_outs):
            if dim_len == 1:
                iters[idx] = IterType.OPAQUE
        if len(outs) > 1:
            expanded_outs = [1] * (len(iters) - len(outs[1])) + list(outs[1])
            for idx, dim_len in enumerate(expanded_outs):
                if dim_len == 1:
                    iters[idx] = IterType.OPAQUE
        in_iters = tuple(
            [tuple(range(len(iters) - len(x), len(iters))) for x in ins])
        out_iters = tuple(
            [tuple(range(len(iters) - len(y), len(iters))) for y in outs])
        return IterSpace(op, iters, in_iters, out_iters)

    @staticmethod
    def get_embedding(op):
        ins, outs = (op.ins, op.outs)
        iters = [IterType.BATCH] + [IterType.OTHER] * (len(ins[0]) - 1) + [
            IterType.OPAQUE, IterType.OTHER
        ]
        in_x = tuple(range(len(ins[0])))
        in_w = tuple(range(len(ins[0]), len(ins[0]) + 2))
        in_iters = (in_x, in_w)
        out = [i for i in range(len(iters)) if i != len(ins[0])]
        out_iters = (tuple(out), )
        return IterSpace(op, iters, in_iters, out_iters)

    @staticmethod
    def get_embedding_bw(op):
        ins, outs = (op.ins, op.outs)
        iters = [IterType.BATCH] + [IterType.OTHER] * (len(ins[1]) - 1) + [
            IterType.OPAQUE, IterType.OTHER
        ]
        in_dy = tuple([i for i in range(len(iters)) if i != len(ins[1])])
        in_x = tuple(range(len(ins[1])))
        out_dw = tuple(range(len(ins[1]), len(ins[1]) + 2))
        in_iters = (in_dy, in_x)
        out_iters = (out_dw, )
        return IterSpace(op, iters, in_iters, out_iters)
