from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from primitiv.tensor cimport Tensor, _Tensor
from primitiv.shape cimport Shape, _Shape
from primitiv.device cimport Device
from primitiv.initializer cimport Initializer, _Initializer


cdef extern from "primitiv/parameter.h" namespace "primitiv":
    cdef cppclass Parameter:
        Parameter(Parameter &&src)
        Parameter() except +
        Parameter(const string &name, const Shape &shape, Device &device) except +
        Parameter(const string &name, const Shape &shape, const vector[float] &value, Device &device) except +
        Parameter(const string &name, const Shape &shape, const Initializer &init, Device &device) except +
        bool valid() except +
        void reset_value(const vector[float] &value) except +
        void reset_value(const Initializer &init) except +
        void reset_gradient() except +
        void add_stats(const string &name, const Shape &shape) except +
        bool has_stats(const string &name) except +
        const string &name() except +
        const Shape &shape() except +
        Device &device() except +
        Tensor &value() except +
        Tensor &gradient() except +
        Tensor &stats(const string &name) except +
        void save(const string &path, bool with_stats) except +


cdef extern from "parameter_load_wrapper.h" namespace "python_primitiv":
    Parameter* Parameter_load(const string &path, bool with_stats, Device &device) except +


cdef class _Parameter:
    cdef Parameter *wrapped
    cdef Parameter *wrapped_newed


cdef inline _Parameter wrapParameter(Parameter *wrapped) except +:
    cdef _Parameter parameter = _Parameter.__new__(_Parameter)
    parameter.wrapped = wrapped
    return parameter


cdef inline _Parameter wrapParameterWithNew(Parameter *wrapped) except +:
    cdef _Parameter parameter = _Parameter.__new__(_Parameter)
    parameter.wrapped = wrapped
    parameter.wrapped_newed = wrapped
    return parameter
