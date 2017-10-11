from primitiv._device cimport CppDevice, _Device
from primitiv._graph cimport CppGraph, _Graph


cdef extern from "primitiv/default_scope.h" namespace "primitiv":
    cdef cppclass CppDefaultScope "primitiv::DefaultScope" [T]:
        CppDefaultScope() except +
        CppDefaultScope(T &obj) except +


cdef extern from "default_scope_wrapper.h" namespace "python_primitiv":
    cdef CppDevice &DefaultScopeDevice_get() except +
    cdef size_t DefaultScopeDevice_size() except +
    cdef CppGraph &DefaultScopeGraph_get() except +
    cdef size_t DefaultScopeGraph_size() except +


cdef class _DefaultScopeDevice:
    cdef CppDefaultScope[CppDevice] *wrapped
    cdef CppDefaultScope[CppDevice] *wrapped_newed
    cdef _Device obj


cdef class _DefaultScopeGraph:
    cdef CppDefaultScope[CppGraph] *wrapped
    cdef CppDefaultScope[CppGraph] *wrapped_newed
    cdef _Graph obj
