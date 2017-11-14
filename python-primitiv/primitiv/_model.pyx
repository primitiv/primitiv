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


cdef class _ModelParameter:

    def __init__(self, _Model model):
        # NOTE(vbkaisetsu): It becomes circular reference.
        # We can't know when it will be deleted by the garbage collector.
        # Therefore we hold this instance in a weakref to delete it immediately.
        self.model_ref = weakref.ref(model)

    def __getitem__(self, key):
        cdef vector[string] names
        if isinstance(key, str):
            return _Parameter.get_wrapper(&(<_Model> self.model_ref()).wrapped.get_parameter(pystr_to_cppstr(key)))
        elif isinstance(key, tuple):
            for name in key:
                names.push_back(pystr_to_cppstr(name))
            return _Parameter.get_wrapper(&(<_Model> self.model_ref()).wrapped.get_parameter(names))
        else:
            raise TypeError("Argument 'key' has incorrect type (str or tuple)")


cdef class _ModelSubModel:

    def __init__(self, _Model model):
        # NOTE(vbkaisetsu): It becomes circular reference.
        # We can't know when it will be deleted by the garbage collector.
        # Therefore we hold this instance in a weakref to delete it immediately.
        self.model_ref = weakref.ref(model)

    def __getitem__(self, key):
        cdef vector[string] names
        if isinstance(key, str):
            return _Model.get_wrapper(&(<_Model> self.model_ref()).wrapped.get_submodel(pystr_to_cppstr(key)))
        elif isinstance(key, tuple):
            for name in key:
                names.push_back(pystr_to_cppstr(name))
            return _Model.get_wrapper(&(<_Model> self.model_ref()).wrapped.get_submodel(names))
        else:
            raise TypeError("Argument 'key' has incorrect type (str or tuple)")


cdef class _Model:

    def __cinit__(self):
        self.params = _ModelParameter(self)
        self.submodels = _ModelSubModel(self)

    def __init__(self):
        if self.wrapped is not NULL:
            raise TypeError("__init__() has already been called.")
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

    def auto_add_attributes(self):
        param_to_add = {} # type: Dict[_Parameter, str]
        for k, v in self.__dict__.items():
            if not isinstance(v, _Parameter):
                continue
            if v in self.added_parameters:
                continue
            if v in param_to_add:
                raise ValueError("A parameter assigned to %s is also assigned to %s." % (k, param_to_add[v]))
            param_to_add[v] = k
        model_to_add = {} # type: Dict[_Parameter, str]
        for k, v in self.__dict__.items():
            if not isinstance(v, _Model):
                continue
            if v in self.added_submodels:
                continue
            if v in model_to_add:
                raise ValueError("A parameter assigned to %s is also assigned to %s." % (k, model_to_add[v]))
            model_to_add[v] = k
        for k, v in param_to_add.items():
            self.add_parameter(v, k)
        for k, v in model_to_add.items():
            self.add_submodel(v, k)

    # NOTE(vbkaisetsu):
    # get_parameter is replaced with `params` variable.

    # NOTE(vbkaisetsu):
    # get_submodel is replaced with `submodels` variable.

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
