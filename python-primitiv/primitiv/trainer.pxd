from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp cimport bool
from libcpp.memory cimport shared_ptr

from primitiv.device cimport Device
from primitiv.shape cimport Shape
from primitiv.parameter cimport Parameter, _Parameter


cdef extern from "primitiv/trainer.h" namespace "primitiv":
    cdef cppclass Trainer:
        Trainer(Trainer &&) except +
        Trainer() except +
        void save(const string &path) except +
        string name() except +
        unsigned get_epoch() except +
        void set_epoch(unsigned epoch) except +
        float get_learning_rate_scaling() except +
        void set_learning_rate_scaling(float scale) except +
        float get_weight_decay() except +
        void set_weight_decay(float strength) except +
        float get_gradient_clipping() except +
        void set_gradient_clipping(float threshold) except +
        void add_parameter(Parameter &param) except +
        void reset_gradients() except +
        void update() except +
        void set_configs_by_file(const string &path) except +


cdef extern from "primitiv/trainer.h" namespace "primitiv::Trainer":
    string detect_name(const string &path) except +


cdef class _Trainer:
    cdef Trainer *wrapped
    cdef Trainer *wrapped_newed


cdef inline _Trainer wrapTrainer(Trainer *wrapped) except +:
    cdef _Trainer trainer = _Trainer.__new__(_Trainer)
    trainer.wrapped = wrapped
    return trainer
