from primitiv.tensor cimport _Tensor


cdef class _Constant(_Initializer):
    def __cinit__(self, float k):
        self.wrapped = new Constant(k)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef Constant *temp
        if self.wrapped is not NULL:
            temp = <Constant*> self.wrapped
            del temp
            self.wrapped = NULL

    def apply(self, _Tensor x):
        (<Constant*> self.wrapped).apply(x.wrapped)
        return


cdef class _Uniform(_Initializer):
    def __cinit__(self, float lower, float upper):
        self.wrapped = new Uniform(lower, upper)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef Uniform *temp
        if self.wrapped is not NULL:
            temp = <Uniform*> self.wrapped
            del temp
            self.wrapped = NULL

    def apply(self, _Tensor x):
        (<Uniform*> self.wrapped).apply(x.wrapped)
        return


cdef class _Normal(_Initializer):
    def __cinit__(self, float mean, float sd):
        self.wrapped = new Normal(mean, sd)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef Normal *temp
        if self.wrapped is not NULL:
            temp = <Normal*> self.wrapped
            del temp
            self.wrapped = NULL

    def apply(self, _Tensor x):
        (<Normal*> self.wrapped).apply(x.wrapped)
        return


cdef class _Identity(_Initializer):
    def __cinit__(self):
        self.wrapped = new Identity()
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef Identity *temp
        if self.wrapped is not NULL:
            temp = <Identity*> self.wrapped
            del temp
            self.wrapped = NULL

    def apply(self, _Tensor x):
        (<Identity*> self.wrapped).apply(x.wrapped)
        return


cdef class _XavierUniform(_Initializer):
    def __cinit__(self, scale = 1.0):
        self.wrapped = new XavierUniform(scale)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef XavierUniform *temp
        if self.wrapped is not NULL:
            temp = <XavierUniform*> self.wrapped
            del temp
            self.wrapped = NULL

    def apply(self, _Tensor x):
        (<XavierUniform*> self.wrapped).apply(x.wrapped)
        return


cdef class _XavierNormal(_Initializer):
    def __cinit__(self, float scale = 1.0):
        self.wrapped = new XavierNormal(scale)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef XavierNormal *temp
        if self.wrapped is not NULL:
            temp = <XavierNormal*> self.wrapped
            del temp
            self.wrapped = NULL

    def apply(self, _Tensor x):
        (<XavierNormal*> self.wrapped).apply(x.wrapped)
        return
