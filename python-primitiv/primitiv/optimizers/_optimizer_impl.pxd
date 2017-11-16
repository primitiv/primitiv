from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

from primitiv._device cimport CppDevice
from primitiv._optimizer cimport CppOptimizer, _Optimizer


cdef extern from "primitiv/optimizer_impl.h":
    cdef cppclass CppSGD "primitiv::optimizers::SGD" (CppOptimizer):
        CppSGD(float eta)
        float eta()

    cdef cppclass CppMomentumSGD "primitiv::optimizers::MomentumSGD" (CppOptimizer):
        CppMomentumSGD(float eta, float momentum)
        float eta()
        float momentum()

    cdef cppclass CppAdaGrad "primitiv::optimizers::AdaGrad" (CppOptimizer):
        CppAdaGrad(float eta, float eps)
        float eta()
        float eps()

    cdef cppclass CppRMSProp "primitiv::optimizers::RMSProp" (CppOptimizer):
        CppRMSProp(float eta, float alpha, float eps)
        float eta()
        float alpha()
        float eps()

    cdef cppclass CppAdaDelta "primitiv::optimizers::AdaDelta" (CppOptimizer):
        CppAdaDelta(float rho, float eps)
        float rho()
        float eps()

    cdef cppclass CppAdam "primitiv::optimizers::Adam" (CppOptimizer):
        CppAdam(float alpha, float beta1, float beta2, float eps)
        float alpha()
        float beta1()
        float beta2()
        float eps()


cdef class _SGD(_Optimizer):
    pass


cdef class _MomentumSGD(_Optimizer):
    pass


cdef class _AdaGrad(_Optimizer):
    pass


cdef class _RMSProp(_Optimizer):
    pass


cdef class _AdaDelta(_Optimizer):
    pass


cdef class _Adam(_Optimizer):
    pass
