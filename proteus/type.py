from enum import Enum


class MapType(Enum):
    DROP = 0
    SHARD = 1
    REPLICATE = 2

    def __str__(self):
        return MapType(self).name

    def __repr__(self):
        return MapType(self).name


class IterType(Enum):
    BATCH = 30
    REDUCE = 31
    OTHER = 32
    OPAQUE = 33
    CONTINUOUS = 34

    def __str__(self):
        return IterType(self).name

    def __repr__(self):
        return IterType(self).name


class DevType(Enum):
    CPU = 10
    GPU = 11


class DataType(Enum):
    Float16 = 20
    Float32 = 21
    Int32 = 22
    Int64 = 23
    Bool = 24


class OpType(Enum):
    Linear = 1000
    LinearBW = 1001
    Conv2d = 1004
    Conv2dBW = 1005
    Pool2d = 1006
    Pool2dBW = 1007
    BatchNorm2d = 1008
    BatchNorm2dBW = 1009
    ReLU = 1010
    ReLUBW = 1011
    Dropout = 1012
    DropoutBW = 1013
    Reshape = 1014
    Permute = 1015
    Matmul = 1022
    MatmulBW = 1023
    CrossEntropy = 1024
    CrossEntropyBW = 1025
    AdaptivePool2d = 1029
    AdaptivePool2dBW = 1030
    Activation = 1031
    ActivationBW = 1032
    Softmax = 1033
    SoftmaxBW = 1034
    Add = 1035
    AddBW = 1036
    Sub = 1037
    SubBW = 1038
    Mul = 1039
    MulBW = 1040
    Div = 1041
    DivBW = 1042
    Sqrt = 1043
    SqrtBW = 1044
    Split = 1045
    Concat = 1046
    LayerNorm = 1047
    LayerNormBW = 1048
    Embedding = 1049
    EmbeddingBW = 1050
    AttentionMask = 1051
    AttentionMaskBW = 1052
    SliceFW = 1053
    SliceBW = 1054
    SGDApply = 1099
    AdamApply = 1100
    Fused = 1300

    def __str__(self):
        return OpType(self).name

    def __repr__(self):
        return OpType(self).name

    @staticmethod
    def is_optimizer(other):
        return other in [OpType.SGDApply, OpType.AdamApply]


class TaskType(Enum):
    TASK_MEM = 100
    TASK_COMP = 101
    TASK_COMM = 102


def size_of_datatype(dt: DataType):
    if dt == DataType.Float16:
        return 2
    elif dt == DataType.Float32:
        return 4
    elif dt == DataType.Int32:
        return 4
    elif dt == DataType.Int64:
        return 8
    elif dt == DataType.Bool:
        return 1
    else:
        assert False, 'Unknown datatype: {}'.format(dt)


def enum_to_int(enum, enum_item):
    for item in enum:
        if (enum_item == item):
            return item.value

    print(enum_item)
    print(enum)
    assert 0, "unknow enum type " + str(enum_item) + " " + str(enum)
    return -1


def int_to_enum(enum, value):
    for item in enum:
        if (item.value == value):
            return item

    assert 0, "unknow enum value " + str(value) + " " + str(enum)


def enum_to_str(enum, enum_item):
    name = enum(enum_item).name
    return name


def str_to_enum(enum, value):
    for item in enum:
        if (item.name == value):
            return item

    assert 0, "unknow enum value " + value + " " + str(enum)
