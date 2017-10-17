from libcpp.vector cimport vector


cdef class _Shape:

    def __init__(self, dims = None, unsigned batch = 1):
        if dims is None:
            self.wrapped = CppShape()
        else:
            self.wrapped = CppShape(<vector[unsigned]> dims, <unsigned> batch)

    def depth(self):
        return self.wrapped.depth()

    def batch(self):
        return self.wrapped.batch()

    def volume(self):
        return self.wrapped.volume()

    def lower_volume(self, unsigned dim):
        return self.wrapped.lower_volume(dim)

    def size(self):
        return self.wrapped.size()

    def __str__(self):
        return self.wrapped.to_string().decode("utf-8")

    def __getitem__(self, unsigned i):
        return self.wrapped[i]

    def __eq__(_Shape self, _Shape rhs):
        return self.wrapped == rhs.wrapped

    def __ne__(_Shape self, _Shape rhs):
        return self.wrapped != rhs.wrapped

    def has_batch(self):
        return self.wrapped.has_batch()

    def has_compatible_batch(self, _Shape rhs):
        return self.wrapped.has_compatible_batch(rhs.wrapped)

    def is_scalar(self):
        return self.wrapped.is_scalar()

    def is_row_vector(self):
        return self.wrapped.is_row_vector()

    def is_matrix(self):
        return self.wrapped.is_matrix()

    def has_same_dims(self, _Shape rhs):
        return self.wrapped.has_same_dims(rhs.wrapped)

    def has_same_loo_dims(self, _Shape rhs, unsigned dim):
        return self.wrapped.has_same_loo_dims(rhs.wrapped, dim)

    def resize_dim(self, unsigned dim, unsigned m):
        self.wrapped.resize_dim(dim, m)
        return self

    def resize_batch(self, unsigned batch):
        self.wrapped.resize_batch(batch)
        return self

    def update_dim(self, unsigned dim, unsigned m):
        self.wrapped.update_dim(dim, m)

    def update_batch(self, unsigned batch):
        self.wrapped.update_batch(batch)
