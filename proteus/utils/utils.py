import numpy as np


def get_strides(parts):
    strides = [1 for _ in parts]
    for j in range(len(parts) - 2, -1, -1):
        strides[j] = strides[j + 1] * parts[j + 1]
    return tuple(strides)


def flat_to_coordinate(flat_idx, strides):
    coordinate = []
    for s in strides:
        coordinate.append(flat_idx // s)
        flat_idx = flat_idx % s
    return tuple(coordinate)


def coordinate_to_flat(coordinate, strides):
    flat_idx = sum([i * j for i, j in zip(coordinate, strides)])
    return flat_idx


def divide_into_n_parts(my_list, n):
    idx_pairs = []
    for i in range(n):
        extent = len(my_list) // n
        start = i * extent
        if len(my_list) % n > i:
            start += i
            extent += 1
        else:
            start += len(my_list) % n
        idx_pairs.append((start, start + extent))
    return idx_pairs


def get_dst_part_groups(src_parts, src_strides, dst_parts, dst_strides):
    step = []
    for s, d in zip(src_parts, dst_parts):
        if s < d or s % d != 0:
            return None
        step.append(s // d)

    group = {i: [] for i in range(np.prod(dst_parts))}
    for idx in range(np.prod(src_parts)):
        coord = flat_to_coordinate(idx, src_strides)
        src_coord = []
        for i in range(len(coord)):
            src_coord.append(coord[i] // step[i])
        flat_idx = coordinate_to_flat(src_coord, dst_strides)
        group[flat_idx].append(idx)
    return group
