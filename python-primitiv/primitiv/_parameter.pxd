from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from primitiv._tensor cimport CppTensor, _Tensor
from primitiv._shape cimport CppShape, _Shape
from primitiv._device cimport CppDevice
from primitiv._initializer cimport CppInitializer, _Initializer


cdef extern from "primitiv/parameter.h":
    cdef cppclass CppParameter "primitiv::Parameter":
        CppParameter(CppParameter &&src)
        CppParameter() except +
        CppParameter(const string &name, const CppShape &shape, CppDevice &device) except +
        CppParameter(const string &name, const CppShape &shape, const vector[float] &value, CppDevice &device) except +
        CppParameter(const string &name, const CppShape &shape, const CppInitializer &init, CppDevice &device) except +
        bool valid() except +
        void reset_value(const vector[float] &value) except +
        void reset_value(const CppInitializer &init) except +
        void reset_gradient() except +
        void add_stats(const string &name, const CppShape &shape) except +
        bool has_stats(const string &name) except +
        const string &name() except +
        const CppShape &shape() except +
        CppDevice &device() except +
        CppTensor &value() except +
        CppTensor &gradient() except +
        CppTensor &stats(const string &name) except +
        void save(const string &path, bool with_stats) except +


cdef extern from "parameter_load_wrapper.h" namespace "python_primitiv":
    CppParameter* Parameter_load(const string &path, bool with_stats, CppDevice &device) except +


cdef class _Parameter:
    cdef CppParameter *wrapped
    cdef CppParameter *wrapped_newed


cdef inline _Parameter wrapParameter(CppParameter *wrapped) except +:
    cdef _Parameter parameter = _Parameter.__new__(_Parameter)
    parameter.wrapped = wrapped
    return parameter


cdef inline _Parameter wrapParameterWithNew(CppParameter *wrapped) except +:
    cdef _Parameter parameter = _Parameter.__new__(_Parameter)
    parameter.wrapped = wrapped
    parameter.wrapped_newed = wrapped
    return parameter
