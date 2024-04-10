import numpy as np
from math import ceil


class CacheModel:
    dp_cache = {}


def mem_visit_elements(ins, outs):
    nele = 0
    for inshp in ins:
        nele += np.prod(inshp)
    for outshp in outs:
        nele += np.prod(outshp)
    return nele


def estimate_dot_product(M, N, K, dev):
    if (M, K, N) in CacheModel.dp_cache:
        return CacheModel.dp_cache[(M, K, N)]

    arith_intensity = 2 * M * N * K / (4 * (M * N + M * K + N * K))
    if arith_intensity > dev.ops_byte:
        time = 2 * M * N * K / (10**6) / dev.gflops
    else:
        time = 4 * (M * N + M * K + N * K) / (10**6) / dev.mem_bw + 0.005
    CacheModel.dp_cache[(M, K, N)] = time
    return time


def estimate_elementwise(nelements, dev):
    return max(nelements * 4 / (10**6) / dev.mem_bw, 0.01)


def roofline_linear(op, dev):
    # (x, w, b) -> (y)
    ins, outs = op.ins, op.outs
    N = np.prod(ins[0][:-1])
    time = estimate_dot_product(outs[0][-1], N, ins[0][-1], dev)
    if len(ins) > 2:
        time += estimate_elementwise(np.prod(outs[0]), dev)
    return time


def roofline_linear_bw(op, dev):
    # (dy, x, w) -> (dx, dw, db)
    ins, outs = op.ins, op.outs
    N = np.prod(ins[0][:-1])
    time = 0
    if outs[0] == ins[1]:
        time += estimate_dot_product(outs[0][-1], N, ins[0][-1], dev)  # dx
    time += estimate_dot_product(outs[0][-1], ins[0][-1], N, dev)  # dw
    if len(outs[-1]) == 1:
        time += estimate_elementwise(np.prod(ins[0]), dev)
    return time


def roofline_matmul(op, dev):
    # (x1, x2) -> (y)
    ins, outs = op.ins, op.outs
    N = np.prod(outs[0][:-1])
    time = estimate_dot_product(outs[0][-1], N, ins[0][-1], dev)
    return time


def roofline_matmul_bw(op, dev):
    # (dy, x1, x2) -> (dx1, dx2)
    ins, outs = op.ins, op.outs
    N = np.prod(ins[0][:-1])
    time = estimate_dot_product(outs[0][-1], N, ins[0][-1], dev)
    time += estimate_dot_product(outs[0][-1], ins[0][-1], N, dev)
    return time


def roofline_conv2d(op, dev):
    # (x, w, b) -> (y)
    ins, outs = op.ins, op.outs
    M = ins[0][0] * np.prod(outs[0][-2:])
    K = np.prod(ins[1][1:])
    time = estimate_dot_product(M, ins[1][0], K, dev)
    if len(ins) > 2:
        time += estimate_elementwise(np.prod(outs[0]), dev)
    return time


def roofline_conv2d_bw(op, dev):
    # (dy, x, w) -> (dx, dw, db)
    ins, outs = op.ins, op.outs
    M = ins[0][0] * np.prod(ins[0][-2:])
    K = np.prod(ins[2][1:])
    time = 0
    if outs[0] == ins[1]:
        time += estimate_dot_product(K, M, ins[0][1], dev)  # dx
    time += estimate_dot_product(K, ins[0][1], M, dev)  # dw
    if len(outs[-1]) == 1:
        time += estimate_elementwise(np.prod(ins[0]), dev)
    return time


def roofline_embedding(op, dev):
    ins, outs = op.ins, op.outs
    nelements = min(np.prod(ins[0]), ins[1][0]) * ins[1][1] + np.prod(
        ins[0]) + np.prod(outs[0])
    time = estimate_elementwise(nelements, dev)
    return time


def roofline_embedding_bw(op, dev):
    ins, outs = op.ins, op.outs
    nelements = np.prod(ins[0]) + np.prod(
        ins[1]) + min(np.prod(ins[1]), outs[0][0]) * outs[0][1]
    time = estimate_elementwise(nelements, dev)
    return time


def roofline_mem_bound(op, dev):
    ins, outs = op.ins, op.outs
    nelements = mem_visit_elements(ins, outs)
    time = estimate_elementwise(nelements, dev)
    return time
