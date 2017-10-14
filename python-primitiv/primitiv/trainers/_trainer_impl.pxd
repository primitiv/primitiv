from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

from primitiv._device cimport CppDevice
from primitiv._trainer cimport CppTrainer, _Trainer


cdef extern from "primitiv/trainer_impl.h" namespace "primitiv::trainers":
    cdef cppclass CppSGD "primitiv::trainers::SGD" (CppTrainer):
        CppSGD(const float eta)
        void get_configs(unordered_map[string, unsigned] &uint_configs, unordered_map[string, float] &float_configs) except +
        void set_configs(const unordered_map[string, unsigned] &uint_configs, const unordered_map[string, float] &float_configs) except +
        float eta()

    cdef cppclass CppAdaGrad "primitiv::trainers::AdaGrad" (CppTrainer):
        CppAdaGrad(float eta, float eps)
        void get_configs(unordered_map[string, unsigned] &uint_configs, unordered_map[string, float] &float_configs) except +
        void set_configs(const unordered_map[string, unsigned] &uint_configs, const unordered_map[string, float] &float_configs) except +
        float eta()
        float eps()

    cdef cppclass CppAdam "primitiv::trainers::Adam" (CppTrainer):
        CppAdam(float alpha, float beta1, float beta2, float eps)
        void get_configs(unordered_map[string, unsigned] &uint_configs, unordered_map[string, float] &float_configs) except +
        void set_configs(const unordered_map[string, unsigned] &uint_configs, const unordered_map[string, float] &float_configs) except +
        float alpha()
        float beta1()
        float beta2()
        float eps()


cdef class _SGD(_Trainer):
    pass


cdef class _AdaGrad(_Trainer):
    pass


cdef class _Adam(_Trainer):
    pass
