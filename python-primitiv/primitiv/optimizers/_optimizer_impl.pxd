from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

from primitiv._device cimport CppDevice
from primitiv._optimizer cimport CppOptimizer, Optimizer


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


cdef class SGD(Optimizer):
    pass


cdef class MomentumSGD(Optimizer):
    pass


cdef class AdaGrad(Optimizer):
    pass


cdef class RMSProp(Optimizer):
    pass


cdef class AdaDelta(Optimizer):
    pass


cdef class Adam(Optimizer):
    pass
