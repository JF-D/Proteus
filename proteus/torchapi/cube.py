class DeviceCube(object):
    '''A (PP, DP, MP)-shape Device Model.
    '''
    def __init__(self, devices, cube):
        super().__init__()
        self.devices = devices
        self.cube = cube

    @property
    def pp_ndims(self):
        return len(self.cube)

    def pp_mesh(self, idx):
        return self.cube[idx]
