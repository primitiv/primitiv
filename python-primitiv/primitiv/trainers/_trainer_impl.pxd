from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

from primitiv._device cimport Device
from primitiv._trainer cimport Trainer, _Trainer


cdef extern from "primitiv/trainer_impl.h" namespace "primitiv::trainers":
    cdef cppclass SGD(Trainer):
        SGD(const float eta)
        void get_configs(unordered_map[string, unsigned] &uint_configs, unordered_map[string, float] &float_configs) except +
        void set_configs(const unordered_map[string, unsigned] &uint_configs, const unordered_map[string, float] &float_configs) except +
        float eta()

    cdef cppclass Adam(Trainer):
        Adam(float alpha, float beta1, float beta2, float eps)
        void get_configs(unordered_map[string, unsigned] &uint_configs, unordered_map[string, float] &float_configs) except +
        void set_configs(const unordered_map[string, unsigned] &uint_configs, const unordered_map[string, float] &float_configs) except +
        float alpha()
        float beta1()
        float beta2()
        float eps()


cdef class _SGD(_Trainer):
    pass


cdef class _Adam(_Trainer):
    pass
