from libcpp.string cimport string


cdef class _SGD(_Trainer):

    def __init__(self, float eta = 0.1):
        if self.wrapped is not NULL:
            raise MemoryError()
        self.wrapped = new CppSGD(eta)

    def eta(self):
        return (<CppSGD*> self.wrapped).eta()


cdef class _MomentumSGD(_Trainer):

    def __init__(self, float eta = 0.01, float momentum = 0.9):
        if self.wrapped is not NULL:
            raise MemoryError()
        self.wrapped = new CppMomentumSGD(eta, momentum)

    def eta(self):
        return (<CppMomentumSGD*> self.wrapped).eta()

    def momentum(self):
        return (<CppMomentumSGD*> self.wrapped).momentum()


cdef class _AdaGrad(_Trainer):

    def __init__(self, float eta = 0.001, float eps = 1e-8):
        if self.wrapped is not NULL:
            raise MemoryError()
        self.wrapped = new CppAdaGrad(eta, eps)

    def eta(self):
        return (<CppAdaGrad*> self.wrapped).eta()

    def eps(self):
        return (<CppAdaGrad*> self.wrapped).eps()


cdef class _RMSProp(_Trainer):

    def __init__(self, float eta = 0.01, float alpha = 0.9, float eps = 1e-8):
        if self.wrapped is not NULL:
            raise MemoryError()
        self.wrapped = new CppRMSProp(eta, alpha, eps)

    def eta(self):
        return (<CppRMSProp*> self.wrapped).eta()

    def alpha(self):
        return (<CppRMSProp*> self.wrapped).alpha()

    def eps(self):
        return (<CppRMSProp*> self.wrapped).eps()


cdef class _AdaDelta(_Trainer):

    def __init__(self, float rho = 0.95, float eps = 1e-6):
        if self.wrapped is not NULL:
            raise MemoryError()
        self.wrapped = new CppAdaDelta(rho, eps)

    def rho(self):
        return (<CppAdaDelta*> self.wrapped).rho()

    def eps(self):
        return (<CppAdaDelta*> self.wrapped).eps()


cdef class _Adam(_Trainer):

    def __init__(self, float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8):
        if self.wrapped is not NULL:
            raise MemoryError()
        self.wrapped = new CppAdam(alpha, beta1, beta2, eps)

    def alpha(self):
        return (<CppAdam*> self.wrapped).alpha()

    def beta1(self):
        return (<CppAdam*> self.wrapped).beta1()

    def beta2(self):
        return (<CppAdam*> self.wrapped).beta2()

    def eps(self):
        return (<CppAdam*> self.wrapped).eps()
