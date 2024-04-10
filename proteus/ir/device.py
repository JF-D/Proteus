import numpy as np
from proteus import DevType, enum_to_str
from proteus.simulator.cost_model import OpCostModel


class Device(object):

    def __init__(self,
                 id: int,
                 type: DevType,
                 memory: int,
                 gflops: int = 15700,
                 mem_bw: int = 900):
        super().__init__()
        self.type = type
        self.memory = memory
        self.gflops = gflops
        self.mem_bw = mem_bw
        self.id = id

        self._ops_byte = self.gflops / self.mem_bw

    @property
    def ops_byte(self):
        return self._ops_byte

    def set_node(self, node_id):
        self.node_id = node_id

    def __repr__(self):
        return f'{enum_to_str(DevType, self.type)}:{self.id}'


class DeviceTopo(object):

    def __init__(self, devices, bandwidth):
        super().__init__()
        self.devices = devices
        self.bandwidth = bandwidth

        self.nodes = {}
        for dev in self.devices:
            if dev.node_id not in self.nodes:
                self.nodes[dev.node_id] = []
            self.nodes[dev.node_id].append(dev)

    @property
    def ndevs(self):
        return len(self.devices)

    def dev(self, id_or_name):
        if isinstance(id_or_name, str):
            # name: gpu or cpu
            return id_or_name
        else:
            # id: gpu rank in current DeviceTopo
            return self.devices[id_or_name]

    def bw(self, src, dst):
        # i, j: rank in current DeviceTopo
        if isinstance(src, int) and isinstance(dst, int):
            return self.bandwidth[src][dst]

        src_type, src_id = src.split(':')
        dst_type, dst_id = dst.split(':')
        if src_type == 'gpu' and dst_type == 'gpu':
            return self.bandwidth[int(src_id)][int(dst_id)]
        else:
            return min(10, self.bandwidth[int(src_id)][int(dst_id)])

    # >>> device mesh api
    def create_sub_mesh(self, devs=None, nodes=None):
        assert devs is not None or nodes is not None
        if devs is None:
            ranks = []
            devices, bandwidth = [], []
            for node_id in nodes:
                for dev in self.nodes[node_id]:
                    ranks.append(self.devices.index(dev))
                    devices.append(dev)
            for i in ranks:
                bws = []
                for j in ranks:
                    bws.append(self.bw(i, j))
                bandwidth.append(bws)
            return DeviceTopo(devices, bandwidth)
        elif nodes is None:
            devices, bandwidth = [], []
            for dev_id in devs:
                devices.append(self.devices[dev_id])
            for i in devs:
                bws = []
                for j in devs:
                    bws.append(self.bw(i, j))
                bandwidth.append(bws)
            return DeviceTopo(devices, bandwidth)
        else:
            raise NotImplementedError

    def mesh_convert(self, devs, shape, type):
        mesh = []
        for dev_id in devs:
            # device: 'gpu:id' or 'cpu:id'
            mesh.append(f'{type}:{dev_id}')
        return np.array(mesh).reshape(shape)

    def map_devs(self, devs, node_id=None):
        map_dev = []
        for dev_id in devs:
            if node_id is None:
                dev = self.dev(dev_id)
            else:
                dev = self.nodes[node_id][dev_id]
            map_dev.append(dev.id)
        return map_dev

    def make_mesh(self, mesh, type='gpu'):
        mesh = np.array(mesh)
        mesh_shape = mesh.shape
        mesh = self.map_devs(mesh.reshape(-1))
        return self.mesh_convert(mesh, mesh_shape, type)

    def make_node_mesh(self, node_id, mesh, type='gpu'):
        mesh = np.array(mesh)
        mesh_shape = mesh.shape
        mesh = self.map_devs(mesh.reshape(-1), node_id=node_id)
        return self.mesh_convert(mesh, mesh_shape, type)

    def create_mesh(self, mesh_shape, type='gpu'):
        mesh = []
        for dev in self.devices:
            mesh.append(dev.id)
        return self.mesh_convert(mesh, mesh_shape, type)

    def create_node_mesh(self, node_id, mesh_shape, type='gpu'):
        mesh = []
        for dev in self.nodes[node_id]:
            mesh.append(dev.id)
        return self.mesh_convert(mesh, mesh_shape, type)


def make_device_list(ids, type, memory, gflops, mem_bw):
    devs = []
    for idx in ids:
        dev = Device(idx, type, memory, gflops, mem_bw)
        devs.append(dev)
    return devs


class Cluster:

    def __init__(self,
                 topo_file,
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
        self.topo_file = topo_file
        self.n_node = n_node
        self.n_gpu_per_node = n_gpu_per_node
        self.ngpus = self.n_node * self.n_gpu_per_node
        self.gpu_gflops = gpu_gflops
        self.gpu_memory = gpu_memory
        self.gpu_memory_bw = gpu_memory_bw
        self.n_cpu_per_node = n_cpu_per_node
        self.cpu_gflops = cpu_gflops
        self.cpu_memory = cpu_memory
        self.cpu_memory_bw = cpu_memory_bw
        self.gpu_to_gpu = gpu_to_gpu
        self.cpu_to_gpu = cpu_to_gpu
        self.cpu_to_cpu = cpu_to_cpu
        self.inter_node = inter_node
        if 'v100' in self.topo_file:
            self.nvlink = True
        else:
            self.nvlink = False

        self.build_dev_topo()
        # ref to cluster
        OpCostModel.cluster = self

    def build_dev_topo(self):
        devices = []
        gpus = []
        for n in range(self.n_node):
            for g in range(self.n_gpu_per_node):
                gid = n * self.n_gpu_per_node + g
                gpu = Device(gid, DevType.GPU, self.gpu_memory, self.gpu_gflops,
                             self.gpu_memory_bw)
                gpu.set_node(n)
                devices.append(gpu)
                gpus.append(gpu.id)

        cpus = []
        for n in range(self.n_node):
            for c in range(self.n_cpu_per_node):
                cid = self.n_node * self.n_gpu_per_node + n * self.n_cpu_per_node + c
                cpu = Device(cid, DevType.CPU, self.cpu_memory, self.cpu_gflops,
                             self.cpu_memory_bw)
                cpu.set_node(n)
                # devices.append(cpu) # ignore cpu in devices
                cpus.append(cpu.id)

        bandwidth = []
        for i in range(len(devices)):
            bw = []
            for j in range(len(devices)):
                if i == j:
                    bw.append(800)
                else:
                    if devices[i].node_id != devices[j].node_id:
                        bw.append(self.inter_node)
                    elif devices[i].type == devices[j].type:
                        if devices[i].type == DevType.GPU:
                            bw.append(self.gpu_to_gpu)
                        else:
                            bw.append(self.cpu_to_cpu)
                    else:
                        bw.append(self.cpu_to_gpu)
            bandwidth.append(bw)

        self.dev_topo = DeviceTopo(devices, bandwidth)
        self.gpus = gpus
        self.cpus = cpus
        self.nodes = {'cpus': [], 'gpus': []}
        for n in range(self.n_node):
            for i in range(self.n_cpu_per_node):
                idx = n * self.n_cpu_per_node + i
                self.nodes['cpus'].append(cpus[idx])
            for i in range(self.n_gpu_per_node):
                idx = n * self.n_gpu_per_node + i
                self.nodes['gpus'].append(gpus[idx])
