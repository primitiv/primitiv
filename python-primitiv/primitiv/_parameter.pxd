from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uintptr_t

from primitiv._tensor cimport CppTensor, _Tensor
from primitiv._shape cimport CppShape, _Shape
from primitiv._device cimport CppDevice
from primitiv._initializer cimport CppInitializer, _Initializer


cdef extern from "primitiv/parameter.h":
    cdef cppclass CppParameter "primitiv::Parameter":
        CppParameter(CppParameter &&src) except +
        CppParameter() except +
        CppParameter(const CppShape &shape, CppDevice &device) except +
        CppParameter(const CppShape &shape, const vector[float] &value, CppDevice &device) except +
        CppParameter(const CppShape &shape, const CppInitializer &init, CppDevice &device) except +
        bool valid() except +
        void reset_value(const vector[float] &value) except +
        void reset_value(const CppInitializer &init) except +
        void reset_gradient() except +
        void add_stats(const string &name, const CppShape &shape) except +
        bool has_stats(const string &name) except +
        const CppShape &shape() except +
        CppDevice &device() except +
        CppTensor &value() except +
        CppTensor &gradient() except +
        CppTensor &stats(const string &name) except +
        void save(const string &path, bool with_stats) except +


cdef extern from "parameter_load_wrapper.h" namespace "python_primitiv":
    CppParameter* Parameter_load(const string &path, bool with_stats, CppDevice &device) except +


cdef class _ParameterStatistics:
    cdef object param_ref


cdef class _Parameter:
    cdef CppParameter *wrapped
    cdef object __weakref__
    cdef readonly _ParameterStatistics stats
    @staticmethod
    cdef void register_wrapper(CppParameter *ptr, _Parameter wrapper)
    @staticmethod
    cdef _Parameter get_wrapper(CppParameter *ptr)
    @staticmethod
    cdef _Parameter get_wrapper_with_new(CppParameter *ptr)

    # NOTE(vbkaisetsu)
    # _Parameter is always created with `new`, so `del_required` is not used.
    # cdef bool del_required
