from primitiv import trainers as T

cdef class _Trainer:

    def load(self, str path):
        self.wrapped.load(path.encode("utf-8"))
        return

    def save(self, str path):
        self.wrapped.save(path.encode("utf-8"))
        return

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

    def __copy__(self):
        raise NotImplementedError(type(self).__name__ + " does not support `__copy__` for now.")

    def __deepcopy__(self, memo):
        raise NotImplementedError(type(self).__name__ + " does not support `__deepcopy__` for now.")
