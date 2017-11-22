from libcpp.pair cimport pair
from libc.stdint cimport uintptr_t

from primitiv._device cimport _Device
from primitiv._parameter cimport _Parameter
from primitiv.config cimport pystr_to_cppstr, cppstr_to_pystr

from weakref import WeakValueDictionary
import weakref

# NOTE(vbkaisetsu):
# This is used for holding python instances related to C++.
# Without this variable, python instances are always created when C++ class
# instances are returned from functions.
# It means that users can not compare instances by using "is" operator.
cdef object py_primitiv_model_weak_dict = WeakValueDictionary()


cdef class _Model:

    def __cinit__(self):
        self.wrapped = new CppModel()
        _Model.register_wrapper(self.wrapped, self)
        self.added_parameters = []
        self.added_submodels = []

    def __dealloc__(self):
        if self.wrapped is not NULL:
            del self.wrapped
            self.wrapped = NULL

    def load(self, str path, bool with_stats = True, _Device device = None):
        if device is None:
            device = _Device.get_default()
        self.wrapped.load(pystr_to_cppstr(path), with_stats, device.wrapped[0])

    def save(self, str path, bool with_stats = True):
        self.wrapped.save(pystr_to_cppstr(path), with_stats)

    def add_parameter(self, str name, _Parameter param):
        self.wrapped.add_parameter(pystr_to_cppstr(name), param.wrapped[0])
        self.added_parameters.append(param)

    def add_submodel(self, str name, _Model model):
        self.wrapped.add_submodel(pystr_to_cppstr(name), model.wrapped[0])
        self.added_submodels.append(model)

    def add_all_parameters(self):
        for k, v in self.__dict__.items():
            if isinstance(v, _Parameter) and v not in self.added_parameters:
                self.add_parameter(k, v)

    def add_all_submodels(self):
        for k, v in self.__dict__.items():
            if isinstance(v, _Model) and v not in self.added_submodels:
                self.add_submodel(k, v)

    def __getitem__(self, key):
        cdef vector[string] names
        if isinstance(key, str):
            names.push_back(pystr_to_cppstr(key))
        elif isinstance(key, tuple):
            for name in key:
                names.push_back(pystr_to_cppstr(name))
        else:
            raise TypeError("Argument 'key' has incorrect type (str or tuple)")
        try:
            return _Parameter.get_wrapper(&self.wrapped.get_parameter(names))
        except:
            try:
                return _Model.get_wrapper(&self.wrapped.get_submodel(names))
            except:
                # NOTE(vbkaisetsu): DO NOT throw an exception here because
                # error massages generated at above lines will also be shown.
                pass
        raise TypeError("'name' is not a name of neither parameter not submodel")

    def get_all_parameters(self):
        cdef pair[vector[string], CppParameter*] p
        result = {}
        for p in self.wrapped.get_all_parameters():
            result[tuple(cppstr_to_pystr(s) for s in p.first)] = _Parameter.get_wrapper(p.second)
        return result

    def get_trainable_parameters(self):
        cdef pair[vector[string], CppParameter*] p
        result = {}
        for p in self.wrapped.get_trainable_parameters():
            result[tuple(cppstr_to_pystr(s) for s in p.first)] = _Parameter.get_wrapper(p.second)
        return result

    @staticmethod
    cdef void register_wrapper(CppModel *ptr, _Model wrapper):
        if <uintptr_t> ptr in py_primitiv_model_weak_dict:
            raise ValueError("Attempted to register the same C++ object twice.")
        py_primitiv_model_weak_dict[<uintptr_t> ptr] = wrapper

    @staticmethod
    cdef _Model get_wrapper(CppModel *ptr):
        return py_primitiv_model_weak_dict[<uintptr_t> ptr]
