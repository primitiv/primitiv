from libcpp.string cimport string


cdef class _SGD(_Trainer):

    def __init__(self, float eta = 0.1):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        self.wrapped_newed = new CppSGD(eta)
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef CppSGD *temp
        if self.wrapped_newed is not NULL:
            temp = <CppSGD*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

    def name(self):
        return (<CppSGD*> self.wrapped).name()

    def eta(self):
        return (<CppSGD*> self.wrapped).eta()

    def get_configs(self):
        cdef unordered_map[string, unsigned] uint_configs
        cdef unordered_map[string, float] float_configs
        (<CppSGD*> self.wrapped).get_configs(uint_configs, float_configs)
        return (uint_configs, float_configs)

    def set_configs(self, const unordered_map[string, unsigned] uint_configs, const unordered_map[string, float] float_configs):
        (<CppSGD*> self.wrapped).set_configs(uint_configs, float_configs)
        return


cdef class _Adam(_Trainer):

    def __init__(self, float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        self.wrapped_newed = new CppAdam(alpha, beta1, beta2, eps)
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef CppAdam *temp
        if self.wrapped_newed is not NULL:
            temp = <CppAdam*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

    def name(self):
        return (<CppAdam*> self.wrapped).name()

    def alpha(self):
        return (<CppAdam*> self.wrapped).alpha()

    def beta1(self):
        return (<CppAdam*> self.wrapped).beta1()

    def beta2(self):
        return (<CppAdam*> self.wrapped).beta2()

    def eps(self):
        return (<CppAdam*> self.wrapped).eps()

    def get_configs(self):
        cdef unordered_map[string, unsigned] uint_configs
        cdef unordered_map[string, float] float_configs
        (<CppSGD*> self.wrapped).get_configs(uint_configs, float_configs)
        return (uint_configs, float_configs)

    def set_configs(self, unordered_map[string, unsigned] uint_configs, unordered_map[string, float] float_configs):
        (<CppSGD*> self.wrapped).set_configs(uint_configs, float_configs)
        return
