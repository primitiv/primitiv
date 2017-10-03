from libcpp.string cimport string


cdef class _SGD(_Trainer):

    def __cinit__(self, float eta = 0.1):
        self.wrapped = new SGD(eta)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef SGD *temp
        if self.wrapped is not NULL:
            temp = <SGD*> self.wrapped
            del temp
            self.wrapped = NULL

    def name(self):
        return (<SGD*> self.wrapped).name()

    def eta(self):
        return (<SGD*> self.wrapped).eta()


cdef class _Adam(_Trainer):
    def __cinit__(self, float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8):
        self.wrapped = new Adam(alpha, beta1, beta2, eps)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef Adam *temp
        if self.wrapped is not NULL:
            temp = <Adam*> self.wrapped
            del temp
            self.wrapped = NULL

    def name(self):
        return (<Adam*> self.wrapped).name()

    def alpha(self):
        return (<Adam*> self.wrapped).alpha()

    def beta1(self):
        return (<Adam*> self.wrapped).beta1()

    def beta2(self):
        return (<Adam*> self.wrapped).beta2()

    def eps(self):
        return (<Adam*> self.wrapped).eps()
