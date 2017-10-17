from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

from primitiv._device cimport CppDevice
from primitiv._trainer cimport CppTrainer, _Trainer


cdef extern from "primitiv/trainer_impl.h" namespace "primitiv::trainers":
    cdef cppclass CppSGD "primitiv::trainers::SGD" (CppTrainer):
        CppSGD(const float eta)
        float eta()

    cdef cppclass CppMomentumSGD "primitiv::trainers::MomentumSGD" (CppTrainer):
        CppMomentumSGD(const float eta, const float momentum)
        float eta()
        float momentum()

    cdef cppclass CppRMSProp "primitiv::trainers::RMSProp" (CppTrainer):
        CppRMSProp(const float eta, const float alpha, const float eps)
        float eta()
        float alpha()
        float eps()

    cdef cppclass CppAdaGrad "primitiv::trainers::AdaGrad" (CppTrainer):
        CppAdaGrad(float eta, float eps)
        float eta()
        float eps()

    cdef cppclass CppAdam "primitiv::trainers::Adam" (CppTrainer):
        CppAdam(float alpha, float beta1, float beta2, float eps)
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
