from libcpp.string cimport string


cdef class _SGD(_Trainer):

    def __init__(self, float eta = 0.1):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        self.wrapped_newed = new SGD(eta)
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef SGD *temp
        if self.wrapped_newed is not NULL:
            temp = <SGD*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

    def name(self):
        return (<SGD*> self.wrapped).name()

    def eta(self):
        return (<SGD*> self.wrapped).eta()


cdef class _Adam(_Trainer):

    def __init__(self, float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        self.wrapped_newed = new Adam(alpha, beta1, beta2, eps)
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef Adam *temp
        if self.wrapped_newed is not NULL:
            temp = <Adam*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

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
