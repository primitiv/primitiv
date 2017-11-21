from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map as cppmap
from libcpp cimport bool

from primitiv._device cimport CppDevice
from primitiv._parameter cimport CppParameter


cdef extern from "primitiv/model.h":
    cdef cppclass CppModel "primitiv::Model":
        CppModel() except +
        void load(string &path, bool with_stats, CppDevice &device) except +
        void save(string &path, bool with_stats) except +
        void add_parameter(string &name, CppParameter &param) except +
        void add_submodel(string &name, CppModel &model) except +
        CppParameter &get_parameter(string &name) except +
        CppParameter &get_parameter(vector[string] &names) except +
        CppModel &get_submodel(string &name) except +
        CppModel &get_submodel(vector[string] &names) except +
        cppmap[vector[string], CppParameter *] get_all_parameters() except +
        cppmap[vector[string], CppParameter *] get_trainable_parameters() except +


cdef class _Model:
    cdef CppModel *wrapped
    cdef object __weakref__
    cdef object added_parameters
    cdef object added_submodels
    @staticmethod
    cdef void register_wrapper(CppModel *ptr, _Model wrapper)
    @staticmethod
    cdef _Model get_wrapper(CppModel *ptr)

    # NOTE(vbkaisetsu)
    # _Model is always created with `new`, so `del_required` is not used.
    # cdef bool del_required
