from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uintptr_t

from primitiv._tensor cimport CppTensor, Tensor
from primitiv._shape cimport CppShape, Shape
from primitiv._device cimport CppDevice
from primitiv._initializer cimport CppInitializer, Initializer


cdef extern from "primitiv/parameter.h":
    cdef cppclass CppParameter "primitiv::Parameter":
        CppParameter() except +
        CppParameter(const CppShape &shape, const vector[float] &value, CppDevice &device) except +
        CppParameter(const CppShape &shape, const CppInitializer &initializer, CppDevice &device) except +
        void init(const CppShape &shape, const vector[float] &value, CppDevice &device) except +
        void init(const CppShape &shape, const CppInitializer &initializer, CppDevice &device) except +
        void load(const string &path, bool with_stats, CppDevice &device) except +
        void save(const string &path, bool with_stats) except +
        bool valid() except +
        void reset_gradient() except +
        void add_stats(const string &name, const CppShape &shape) except +
        bool has_stats(const string &name) except +
        const CppShape &shape() except +
        CppDevice &device() except +
        CppTensor &value() except +
        CppTensor &gradient() except +
        CppTensor &stats(const string &name) except +


cdef class ParameterStatistics:
    cdef object param_ref


cdef class Parameter:
    cdef CppParameter *wrapped
    cdef object __weakref__
    cdef readonly ParameterStatistics stats
    """A dictionary-like object of the current opotional statistics.

    """
    @staticmethod
    cdef void register_wrapper(CppParameter *ptr, Parameter wrapper)
    @staticmethod
    cdef Parameter get_wrapper(CppParameter *ptr)
    @staticmethod
    cdef Parameter get_wrapper_with_new(CppParameter *ptr)

    # NOTE(vbkaisetsu)
    # Parameter is always created with `new`, so `del_required` is not used.
    # cdef bool del_required
