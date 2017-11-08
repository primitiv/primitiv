from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from primitiv._parameter cimport CppParameter


cdef extern from "primitiv/model.h":
    cdef cppclass CppModel "primitiv::Model":
        CppModel() except +
        void add_parameter(string &name, CppParameter &param) except +
        void add_submodel(string &name, CppModel &model) except +
        CppParameter &get_parameter(string &name) except +
        CppParameter &get_parameter(vector[string] &names) except +
        CppModel &get_submodel(string &name) except +
        CppModel &get_submodel(vector[string] &names) except +


cdef class _ModelParameter:
    cdef object model_ref


cdef class _ModelSubModel:
    cdef object model_ref


cdef class _Model:
    cdef CppModel *wrapped
    cdef object __weakref__
    cdef readonly _ModelParameter params
    cdef readonly _ModelSubModel submodels
    @staticmethod
    cdef void register_wrapper(CppModel *ptr, _Model wrapper)
    @staticmethod
    cdef _Model get_wrapper(CppModel *ptr)

    # NOTE(vbkaisetsu)
    # _Model is always created with `new`, so `del_required` is not used.
    # cdef bool del_required
