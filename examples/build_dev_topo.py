import numpy as np
from proteus import DevType
from proteus.ir import Device, DeviceTopo, make_device_list, Cluster


def build_dev_topo_N(ndev):
    devs = make_device_list(range(ndev), DevType.GPU, 16000, 15700, 900)
    bandwidth = np.ones((ndev, ndev)) * 25
    for i in range(ndev):
        for j in range(ndev):
            if i == j:
                bandwidth[i][j] = 750
            else:
                if i // 8 != j // 8:
                    bandwidth[i][j] = 7

    dev_topo = DeviceTopo(devs, bandwidth)
    return dev_topo


def build_dev_topo_N8():
    devs = make_device_list(range(8), DevType.GPU, 16000, 15700, 900)
    bandwidth = np.ones((8, 8)) * 25
    for i in range(8):
        bandwidth[i][i] = 750
    dev_topo = DeviceTopo(devs, bandwidth)
    return dev_topo


def build_dev_topo_N4():
    devs = make_device_list(range(4), DevType.GPU, 16000, 15700, 900)
    # bandwidth = [[750, 25, 50, 7], [25, 750, 25, 7], [50, 25, 750, 50],
    #              [7, 7, 50, 750]]
    bandwidth = [[750, 25, 25, 25], [25, 750, 25, 25], [25, 25, 750, 25],
                 [25, 25, 25, 750]]
    dev_topo = DeviceTopo(devs, bandwidth)
    return dev_topo


def build_dev_topo_N2():
    devs = make_device_list(range(2), DevType.GPU, 16000, 15700, 900)
    bandwidth = [[750, 25], [25, 750]]
    dev_topo = DeviceTopo(devs, bandwidth)
    return dev_topo


def build_dev_topo_N1():
    g0 = Device(0, DevType.GPU, 16000, 15700, 900)
    bandwidth = [[750]]
    dev_topo = DeviceTopo([g0], bandwidth)
    return dev_topo


func_dict = {
    1: build_dev_topo_N1,
    2: build_dev_topo_N2,
    4: build_dev_topo_N4,
    8: build_dev_topo_N8
}


def build_dev_topo(ndev):
    if ndev in func_dict:
        return func_dict[ndev]()
    return build_dev_topo_N(ndev)


def build_cluster(topo_file,
                  n_node=1,
                  n_gpu_per_node=8,
                  gpu_gflops=15700,
                  gpu_memory=16000,
                  gpu_memory_bw=900,
                  n_cpu_per_node=8,
                  cpu_gflops=1500,
                  cpu_memory=100000,
                  cpu_memory_bw=128,
                  gpu_to_gpu=48,
                  cpu_to_gpu=10,
                  cpu_to_cpu=50,
                  inter_node=7):
    return Cluster(topo_file,
                   n_node=n_node,
                   n_gpu_per_node=n_gpu_per_node,
                   gpu_gflops=gpu_gflops,
                   gpu_memory=gpu_memory,
                   gpu_memory_bw=gpu_memory_bw,
                   n_cpu_per_node=n_cpu_per_node,
                   cpu_gflops=cpu_gflops,
                   cpu_memory=cpu_memory,
                   cpu_memory_bw=cpu_memory_bw,
                   gpu_to_gpu=gpu_to_gpu,
                   cpu_to_gpu=cpu_to_gpu,
                   cpu_to_cpu=cpu_to_cpu,
                   inter_node=inter_node)
