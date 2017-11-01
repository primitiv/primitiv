from primitiv import trainers as T
from primitiv._parameter cimport _Parameter


cdef class _Trainer:

    def __init__(self):
        if self.wrapped is not NULL:
            raise MemoryError()
        self.wrapped = new CppPyTrainer(self)

    # NOTE(vbkaisetsu):
    # This method is also used by child classes implemented in
    # trainers/_trainer_impl.pyx
    def __dealloc__(self):
        if self.wrapped is not NULL:
            del self.wrapped
            self.wrapped = NULL

    @staticmethod
    def load(str path):
        name = _Trainer.detect_name(path);
        if name == "SGD":
            trainer = T.SGD()
        elif name == "MomentumSGD":
            trainer = T.MomentumSGD()
        elif name == "AdaGrad":
            trainer = T.AdaGrad()
        elif name == "RMSProp":
            trainer = T.RMSProp()
        elif name == "AdaDelta":
            trainer = T.AdaDelta()
        elif name == "Adam":
            trainer = T.Adam()
        else:
            raise OSError("Unknown trainer name: %s" % name)
        trainer.set_configs_by_file(path)
        return trainer

    @staticmethod
    def detect_name(str path):
        return CppTrainer_detect_name(path.encode("utf-8")).decode("utf-8")

    def save(self, str path):
        self.wrapped.save(path.encode("utf-8"))
        return

    def name(self):
        return self.wrapped.name().decode("utf-8")

    def get_epoch(self):
        return self.wrapped.get_epoch()

    def set_epoch(self, unsigned epoch):
        self.wrapped.set_epoch(epoch)
        return

    def get_learning_rate_scaling(self):
        return self.wrapped.get_learning_rate_scaling()

    def set_learning_rate_scaling(self, float scale):
        self.wrapped.set_learning_rate_scaling(scale)
        return

    def get_weight_decay(self):
        return self.wrapped.get_weight_decay()

    def set_weight_decay(self, float strength):
        self.wrapped.set_weight_decay(strength)
        return

    def get_gradient_clipping(self):
        return self.wrapped.get_gradient_clipping()

    def set_gradient_clipping(self, float threshold):
        self.wrapped.set_gradient_clipping(threshold)
        return

    def add_parameter(self, _Parameter param):
        self.wrapped.add_parameter(param.wrapped[0])
        return

    def reset_gradients(self):
        self.wrapped.reset_gradients()
        return

    def update(self):
        self.wrapped.update()
        return

    def get_configs(self):
        cdef unordered_map[string, unsigned] uint_configs
        cdef unordered_map[string, float] float_configs
        self.wrapped.get_configs(uint_configs, float_configs)
        return ({k.decode("utf-8"): v for k, v in dict(uint_configs).items()},
                {k.decode("utf-8"): v for k, v in dict(float_configs).items()})

    def set_configs(self, dict uint_configs, dict float_configs):
        self.wrapped.set_configs({k.encode("utf-8"): v for k, v in uint_configs.items()},
                                 {k.encode("utf-8"): v for k, v in float_configs.items()})
        return

    def set_configs_by_file(self, str path):
        self.wrapped.set_configs_by_file(path.encode("utf-8"))
        return

    def __copy__(self):
        raise NotImplementedError(type(self).__name__ + " does not support `__copy__` for now.")

    def __deepcopy__(self, memo):
        raise NotImplementedError(type(self).__name__ + " does not support `__deepcopy__` for now.")


cdef public api int python_primitiv_trainer_name(
                        object self,
                        string *val) except -1:
    # NOTE(vbkaisetsu):
    # `hasattr(self.__class__, "name")` also scans a parent class.
    # We want check that the function is overrided or not.
    if "name" not in self.__class__.__dict__ or not callable(self.name):
        raise NotImplementedError("'name()' is not implemented in '%s'"
                                        % self.__class__.__name__)
    ret_str = self.name()
    val[0] = ret_str.encode("utf-8")
    return 0


cdef public api int python_primitiv_trainer_configure_parameter(
                        object self,
                        CppParameter &param) except -1:
    # NOTE(vbkaisetsu):
    # `hasattr(self.__class__, "configure_parameter")` also scans a parent class.
    # We want check that the function is overrided or not.
    if "configure_parameter" not in self.__class__.__dict__ or not callable(self.configure_parameter):
        raise NotImplementedError("'configure_parameter()' is not implemented in '%s'"
                                        % self.__class__.__name__)
    self.configure_parameter(_Parameter.get_wrapper(&param))
    return 0


cdef public api int python_primitiv_trainer_update_parameter(
                        object self,
                        float scale,
                        CppParameter &param) except -1:
    # NOTE(vbkaisetsu):
    # `hasattr(self.__class__, "update_parameter")` also scans a parent class.
    # We want check that the function is overrided or not.
    if "update_parameter" not in self.__class__.__dict__ or not callable(self.update_parameter):
        raise NotImplementedError("'update_parameter()' is not implemented in '%s'"
                                        % self.__class__.__name__)
    self.update_parameter(scale, _Parameter.get_wrapper(&param))
    return 0


cdef public api int python_primitiv_trainer_get_configs(
                        object self,
                        unordered_map[string, unsigned] &uint_configs,
                        unordered_map[string, float] &float_configs) except -1:
    # NOTE(vbkaisetsu):
    # `hasattr(self.__class__, "get_configs")` also scans a parent class.
    # We want check that the function is overrided or not.
    if "get_configs" not in self.__class__.__dict__ or not callable(self.get_configs):
        raise NotImplementedError("'get_configs()' is not implemented in '%s'"
                                        % self.__class__.__name__)
    uint_configs_tmp, float_configs_tmp = self.get_configs()
    uint_configs.swap({k.encode("utf-8"): v for k, v in uint_configs_tmp.items()})
    float_configs.swap({k.encode("utf-8"): v for k, v in float_configs_tmp.items()})
    return 0


cdef public api int python_primitiv_trainer_set_configs(
                        object self,
                        const unordered_map[string, unsigned] &uint_configs,
                        const unordered_map[string, float] &float_configs) except -1:
    # NOTE(vbkaisetsu):
    # `hasattr(self.__class__, "set_configs")` also scans a parent class.
    # We want check that the function is overrided or not.
    if "set_configs" not in self.__class__.__dict__ or not callable(self.set_configs):
        raise NotImplementedError("'set_configs()' is not implemented in '%s'"
                                        % self.__class__.__name__)
    self.set_configs({k.decode("utf-8"): v for k, v in dict(uint_configs).items()},
                     {k.decode("utf-8"): v for k, v in dict(float_configs).items()})
    return 0
