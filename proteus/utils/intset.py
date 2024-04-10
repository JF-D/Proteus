import random


class IntSet(object):
    """An integer set.

    Integers in [start, end] that divide stride.
    [1, stride, 2*stride, 3*stride, ...]
    """
    def __init__(self, start, end, stride_or_candidates=1):
        super().__init__()
        self.start = start  # generally, start is 1
        self.end = end
        if isinstance(stride_or_candidates, int):
            self.stride = stride_or_candidates
            self._len = end // self.stride - (start - 1) // self.stride
            if self.stride != 1 and self.start == 1:
                self._len += 1
        else:
            self._candidates = stride_or_candidates
            self._len = len(self._candidates)
            self.stride = None

    def __len__(self):
        return self._len

    def get(self, n):
        if self.stride is None:
            return self._candidates[n]
        elif self.stride == 1:
            res = self.start + n * self.stride
        else:
            if self.start != 1:
                res = ((self.start - 1) // self.stride + n + 1) * self.stride
            else:
                res = self.start if n == 0 else n * self.stride
        return res

    @property
    def candidates(self):
        if self.stride is None:
            return tuple(self._candidates)
        datas = []
        for i in range(self._len):
            datas.append(self.get(i))
        return tuple(datas)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self._len:
            raise StopIteration
        else:
            res = self.get(self.idx)
            self.idx += 1
            return res

    def random(self):
        n = random.randint(0, self._len - 1)
        return self.get(n)
