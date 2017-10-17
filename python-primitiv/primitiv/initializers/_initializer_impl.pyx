from primitiv._tensor cimport _Tensor


cdef class _Constant(_Initializer):

    def __init__(self, float k):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        self.wrapped_newed = new CppConstant(k)
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef CppConstant *temp
        if self.wrapped_newed is not NULL:
            temp = <CppConstant*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

    def apply(self, _Tensor x):
        (<CppConstant*> self.wrapped).apply(x.wrapped)
        return


cdef class _Uniform(_Initializer):

    def __init__(self, float lower, float upper):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        self.wrapped_newed = new CppUniform(lower, upper)
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef CppUniform *temp
        if self.wrapped_newed is not NULL:
            temp = <CppUniform*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

    def apply(self, _Tensor x):
        (<CppUniform*> self.wrapped).apply(x.wrapped)
        return


cdef class _Normal(_Initializer):

    def __init__(self, float mean, float sd):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        self.wrapped_newed = new CppNormal(mean, sd)
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef CppNormal *temp
        if self.wrapped_newed is not NULL:
            temp = <CppNormal*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

    def apply(self, _Tensor x):
        (<CppNormal*> self.wrapped).apply(x.wrapped)
        return


cdef class _Identity(_Initializer):

    def __init__(self):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        self.wrapped_newed = new CppIdentity()
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef CppIdentity *temp
        if self.wrapped_newed is not NULL:
            temp = <CppIdentity*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

    def apply(self, _Tensor x):
        (<CppIdentity*> self.wrapped).apply(x.wrapped)
        return


cdef class _XavierUniform(_Initializer):

    def __init__(self, scale = 1.0):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        self.wrapped_newed = new CppXavierUniform(scale)
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef CppXavierUniform *temp
        if self.wrapped_newed is not NULL:
            temp = <CppXavierUniform*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

    def apply(self, _Tensor x):
        (<CppXavierUniform*> self.wrapped).apply(x.wrapped)
        return


cdef class _XavierNormal(_Initializer):

    def __init__(self, float scale = 1.0):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        self.wrapped_newed = new CppXavierNormal(scale)
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef CppXavierNormal *temp
        if self.wrapped_newed is not NULL:
            temp = <CppXavierNormal*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

    def apply(self, _Tensor x):
        (<CppXavierNormal*> self.wrapped).apply(x.wrapped)
        return
