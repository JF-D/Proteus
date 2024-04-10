class ListQueue:
    def __init__(self, comm=False):
        self.elements = []
        self.comm = comm

    def put(self, element):
        if self.comm:
            if element.is_grad_comm:
                self.elements.append(element)
            else:
                find = -1
                for i in range(self.size()):
                    if self.elements[i].is_grad_comm:
                        find = i
                        break
                if find == -1:
                    self.elements.append(element)
                else:
                    if find == 0:
                        self.elements[i].wait_n += 1
                        element.wait_n -= 1
                    self.elements.insert(find, element)
        else:
            self.elements.append(element)

    def get(self):
        # for i in range(len(self.elements)):
        #     if not self.elements[i].is_recomputation:
        #         return self.elements.pop(i)
        return self.elements.pop(0)

    def empty(self):
        return len(self.elements) == 0

    def size(self):
        return len(self.elements)

    def __getitem__(self, key):
        return self.elements[key]

    def __setitem__(self, key, value):
        self.elements[key] = value

    def __repr__(self):
        return '{}'.format(self.elements)
