from primitiv import trainers as T

cdef class _Trainer:

    @staticmethod
    def load(str path):
        name = _Trainer.detect_name(path);
        if name == "SGD":
            trainer = T.SGD()
        elif name == "Adam":
            trainer = T.Adam()
        else:
            raise IOError("Unknown trainer name:", name)
        trainer.set_configs_by_file(path)
        return trainer

    @staticmethod
    def detect_name(str path):
        return detect_name(path.encode("utf-8")).decode("utf-8")

    def save(self, str path):
        self.wrapped.save(path.encode("utf-8"))
        return

    def name(self):
        return self.wrapped.name()

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

    def set_configs_by_file(self, str path):
        self.wrapped.set_configs_by_file(path.encode("utf-8"))
        return
