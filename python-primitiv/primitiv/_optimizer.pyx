from primitiv._model cimport Model
from primitiv._parameter cimport Parameter
from primitiv.config cimport pystr_to_cppstr, cppstr_to_pystr


cdef class Optimizer:
    """Abstract class for parameter optimizers.

    """

    # NOTE(vbkaisetsu):
    # This method should be called in the __init__() method of a
    # custom Optimizer class.
    #
    # Users can define custom optimizers written in Python. This method
    # generates an instance of a helper optimizer called "PyOptimizer" that
    # can call methods implemented in child classes of Optimizer.
    def __init__(self):
        """Creates a new Python Optimizer.

        To create a new optimizer implemented in Python, call the base
        initializer in ``__init__`` function, and define at least four methods:
        ``configure_parameter``, ``update_parameter``, ``get_configs``,
        ``set_configs`` in the sub-class of the ``Optimizer``.

        Example:

            >>> class MyOptimizer(Optimizer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         :::
            ...
            ...     def configure_parameter(self, param):
            ...         :::
            ...
            ...     def update_parameter(self, scale, param):
            ...         :::
            ...
            ...     def get_configs(self):
            ...         :::
            ...         return uint_configs, float_configs
            ...
            ...     def set_configs(self, uint_configs, float_configs):
            ...         :::

        """
        if self.wrapped is not NULL:
            raise TypeError("__init__() has already been called.")
        self.wrapped = new CppPyOptimizer(self)

    # NOTE(vbkaisetsu):
    # This method is also used by child classes implemented in
    # optimizers/_optimizer_impl.pyx
    # Please be careful when you change behavior around pointer of PyOptimizer.
    def __dealloc__(self):
        # NOTE(vbkaisetsu):
        # DO NOT delete C++ instance without checking NULL.
        # __init__() is not guaranteed to be called when an instance is created.
        # e.g. __new__() method, inherited without __init__(), etc.
        if self.wrapped is not NULL:
            del self.wrapped
            self.wrapped = NULL

    def load(self, str path):
        """Loads configurations from a file.

        :param path: Path of the optimizer parameter file.
        :type path: str

        """
        self.wrapped.load(pystr_to_cppstr(path))
        return

    def save(self, str path):
        """Saves current configurations to a file.

        :param path: Path of the file that will store optimizer parameters.
        :type path: str

        """
        self.wrapped.save(pystr_to_cppstr(path))
        return

    def get_epoch(self):
        """Retrieves current epoch.

        :return: Current epoch.
        :rtype: int

        """
        return self.wrapped.get_epoch()

    def set_epoch(self, unsigned epoch):
        """Sets current epoch.

        :param epoch: New epoch.
        :type epoch: int

        """
        self.wrapped.set_epoch(epoch)
        return

    def get_learning_rate_scaling(self):
        """Retrieves current learning rate scaling factor.

        :return: The scaling factor.
        :rtype: float

        """
        return self.wrapped.get_learning_rate_scaling()

    def set_learning_rate_scaling(self, float scale):
        """Sets learning rate scaling factor.

        :param scale: New scaling factor.
        :type scale: float

        Could not set negative values.

        """
        self.wrapped.set_learning_rate_scaling(scale)
        return

    def get_weight_decay(self):
        """Retrieves current L2 decay strength.

        :return: Current L2 decay strength.
        :rtype: float

        """
        return self.wrapped.get_weight_decay()

    def set_weight_decay(self, float strength):
        """Sets L2 decay strength.

        :param strength: New L2 decay strength, or 0 to disable L2 decay.
        :type strength: float

        Could not set negative values.

        """
        self.wrapped.set_weight_decay(strength)
        return

    def get_gradient_clipping(self):
        """Retrieves current gradient clipping threshold.

        :return: Current gradient clipping threshold.
        :rtype: float

        """
        return self.wrapped.get_gradient_clipping()

    def set_gradient_clipping(self, float threshold):
        """Sets gradient clipping threshold.

        :param threshold: New clipping threshold, or 0 to disable gradient clipping.
        :type threshold: float

        Could not set negative values.

        """
        self.wrapped.set_gradient_clipping(threshold)
        return

    def add_parameter(self, Parameter param):
        """Registers a parameter.

        :param param: Parameter to be optimized.
        :type param: primitiv.Parameter

        """
        self.wrapped.add_parameter(param.wrapped[0])
        return

    def add_model(self, Model model):
        """Registers all trainable parameters in a model.

        :param model: Model to be optimized.
        :type model: primitiv.Model

        """
        self.wrapped.add_model(model.wrapped[0])
        return

    def reset_gradients(self):
        """Resets all gradients of registered parameters.

        """
        self.wrapped.reset_gradients()
        return

    def update(self):
        """Updates parameter values.

        """
        self.wrapped.update()
        return

    def get_configs(self):
        """Gathers configuration values.

        :return: Tuple of configurations with ``int`` type and ``float`` type.
        :rtype: tuple[dict[str, int], dict[str, float]]

        """
        cdef unordered_map[string, unsigned] uint_configs
        cdef unordered_map[string, float] float_configs
        self.wrapped.get_configs(uint_configs, float_configs)
        return ({cppstr_to_pystr(k): v for k, v in dict(uint_configs).items()},
                {cppstr_to_pystr(k): v for k, v in dict(float_configs).items()})

    def set_configs(self, dict uint_configs, dict float_configs):
        """Sets configuration values.

        :param uint_configs: Configurations with ``int`` type.
        :type uint_configs: dict[str, int]
        :param float_configs: Configurations with ``float`` type.
        :type float_configs: dict[str, float]

        """
        self.wrapped.set_configs({pystr_to_cppstr(k): v for k, v in uint_configs.items()},
                                 {pystr_to_cppstr(k): v for k, v in float_configs.items()})
        return

    def __copy__(self):
        raise NotImplementedError(type(self).__name__ + " does not support `__copy__` for now.")

    def __deepcopy__(self, memo):
        raise NotImplementedError(type(self).__name__ + " does not support `__deepcopy__` for now.")


cdef public api int python_primitiv_optimizer_configure_parameter(
                        object self,
                        CppParameter &param) except -1:
    # NOTE(vbkaisetsu):
    # `hasattr(self.__class__, "configure_parameter")` also scans a parent class.
    # We want check that the function is overrided or not.
    if "configure_parameter" in self.__class__.__dict__ and callable(self.configure_parameter):
        self.configure_parameter(Parameter.get_wrapper(&param))
        return 0
    raise NotImplementedError("'configure_parameter()' is not implemented in '%s'"
                                        % self.__class__.__name__)


cdef public api int python_primitiv_optimizer_update_parameter(
                        object self,
                        float scale,
                        CppParameter &param) except -1:
    # NOTE(vbkaisetsu):
    # `hasattr(self.__class__, "update_parameter")` also scans a parent class.
    # We want check that the function is overrided or not.
    if "update_parameter" in self.__class__.__dict__ and callable(self.update_parameter):
        self.update_parameter(scale, Parameter.get_wrapper(&param))
        return 0
    raise NotImplementedError("'update_parameter()' is not implemented in '%s'"
                                        % self.__class__.__name__)


cdef public api int python_primitiv_optimizer_get_configs(
                        object self,
                        unordered_map[string, unsigned] &uint_configs,
                        unordered_map[string, float] &float_configs) except -1:
    # NOTE(vbkaisetsu):
    # `hasattr(self.__class__, "get_configs")` also scans a parent class.
    # We want check that the function is overrided or not.
    if "get_configs" in self.__class__.__dict__ and callable(self.get_configs):
        uint_configs_tmp, float_configs_tmp = self.get_configs()
        uint_configs.swap({pystr_to_cppstr(k): v for k, v in uint_configs_tmp.items()})
        float_configs.swap({pystr_to_cppstr(k): v for k, v in float_configs_tmp.items()})
        return 0
    raise NotImplementedError("'get_configs()' is not implemented in '%s'"
                                        % self.__class__.__name__)


cdef public api int python_primitiv_optimizer_set_configs(
                        object self,
                        const unordered_map[string, unsigned] &uint_configs,
                        const unordered_map[string, float] &float_configs) except -1:
    # NOTE(vbkaisetsu):
    # `hasattr(self.__class__, "set_configs")` also scans a parent class.
    # We want check that the function is overrided or not.
    if "set_configs" in self.__class__.__dict__ and callable(self.set_configs):
        self.set_configs({cppstr_to_pystr(k): v for k, v in dict(uint_configs).items()},
                         {cppstr_to_pystr(k): v for k, v in dict(float_configs).items()})
        return 0
    raise NotImplementedError("'set_configs()' is not implemented in '%s'"
                                        % self.__class__.__name__)
