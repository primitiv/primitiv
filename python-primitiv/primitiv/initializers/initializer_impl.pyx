from primitiv.tensor cimport _Tensor


cdef class _Constant(_Initializer):

    def __init__(self, float k):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        self.wrapped_newed = new Constant(k)
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef Constant *temp
        if self.wrapped_newed is not NULL:
            temp = <Constant*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

    def apply(self, _Tensor x):
        (<Constant*> self.wrapped).apply(x.wrapped)
        return


cdef class _Uniform(_Initializer):

    def __init__(self, float lower, float upper):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        self.wrapped_newed = new Uniform(lower, upper)
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef Uniform *temp
        if self.wrapped_newed is not NULL:
            temp = <Uniform*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

    def apply(self, _Tensor x):
        (<Uniform*> self.wrapped).apply(x.wrapped)
        return


cdef class _Normal(_Initializer):

    def __init__(self, float mean, float sd):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        self.wrapped_newed = new Normal(mean, sd)
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef Normal *temp
        if self.wrapped_newed is not NULL:
            temp = <Normal*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

    def apply(self, _Tensor x):
        (<Normal*> self.wrapped).apply(x.wrapped)
        return


cdef class _Identity(_Initializer):

    def __init__(self):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        self.wrapped_newed = new Identity()
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef Identity *temp
        if self.wrapped_newed is not NULL:
            temp = <Identity*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

    def apply(self, _Tensor x):
        (<Identity*> self.wrapped).apply(x.wrapped)
        return


cdef class _XavierUniform(_Initializer):

    def __init__(self, scale = 1.0):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        self.wrapped_newed = new XavierUniform(scale)
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef XavierUniform *temp
        if self.wrapped_newed is not NULL:
            temp = <XavierUniform*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

    def apply(self, _Tensor x):
        (<XavierUniform*> self.wrapped).apply(x.wrapped)
        return


cdef class _XavierNormal(_Initializer):

    def __init__(self, float scale = 1.0):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        self.wrapped_newed = new XavierNormal(scale)
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef XavierNormal *temp
        if self.wrapped_newed is not NULL:
            temp = <XavierNormal*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

    def apply(self, _Tensor x):
        (<XavierNormal*> self.wrapped).apply(x.wrapped)
        return
