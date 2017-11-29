from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp cimport bool
from libcpp.memory cimport shared_ptr

from primitiv._device cimport CppDevice
from primitiv._model cimport CppModel
from primitiv._parameter cimport CppParameter, Parameter
from primitiv._shape cimport CppShape


cdef extern from "primitiv/optimizer.h":
    cdef cppclass CppOptimizer "primitiv::Optimizer":
        CppOptimizer(CppOptimizer &&) except +
        CppOptimizer() except +
        void load(const string &path) except +
        void save(const string &path) except +
        unsigned get_epoch() except +
        void set_epoch(unsigned epoch) except +
        float get_learning_rate_scaling() except +
        void set_learning_rate_scaling(float scale) except +
        float get_weight_decay() except +
        void set_weight_decay(float strength) except +
        float get_gradient_clipping() except +
        void set_gradient_clipping(float threshold) except +
        void add_parameter(CppParameter &param) except +
        void add_model(CppModel &model) except +
        void reset_gradients() except +
        void update() except +
        void get_configs(unordered_map[string, unsigned] &uint_configs, unordered_map[string, float] &float_configs) except +
        void set_configs(const unordered_map[string, unsigned] &uint_configs, const unordered_map[string, float] &float_configs) except +


cdef class Optimizer:
    cdef CppOptimizer *wrapped


cdef extern from "py_optimizer.h":
    cdef cppclass CppPyOptimizer "python_primitiv::PyOptimizer" (CppOptimizer):
        CppPyOptimizer(object obj) except +
