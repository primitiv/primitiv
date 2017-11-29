from libcpp.string cimport string


cdef class SGD(Optimizer):
    """Simple stochastic gradient descent.

    """

    def __init__(self, float eta = 0.1):
        """Creates a new SGD object.

        :param eta: Learning rate.
        :type eta: float

        """
        if self.wrapped is not NULL:
            raise TypeError("__init__() has already been called.")
        self.wrapped = new CppSGD(eta)

    def eta(self):
        """Returns the learning rate.

        :return: Learning rate.
        :rtype: float

        """
        return (<CppSGD*> self.wrapped).eta()


cdef class MomentumSGD(Optimizer):
    """Stochastic gradient descent with momentum.

    """

    def __init__(self, float eta = 0.01, float momentum = 0.9):
        """Creates a new MomentumSGD object.

        :param eta: Learning rate (default: ``0.01``).
        :type eta: float
        :param momentum: Decay factor of the momentum (default: ``0.9``).
        :type momentum: float

        """
        if self.wrapped is not NULL:
            raise TypeError("__init__() has already been called.")
        self.wrapped = new CppMomentumSGD(eta, momentum)

    def eta(self):
        """Returns the learning rate.

        :return: Learning rate.
        :rtype: float

        """
        return (<CppMomentumSGD*> self.wrapped).eta()

    def momentum(self):
        """Returns the hyperparameter momentum.

        :return: The value of momentum.
        :rtype: float

        """
        return (<CppMomentumSGD*> self.wrapped).momentum()


cdef class AdaGrad(Optimizer):
    """AdaGrad optimizer.

    """

    def __init__(self, float eta = 0.001, float eps = 1e-8):
        """Creates a new AdaGrad object.

        :param eta: Learning rate (default: ``0.001``).
        :type eta: float
        :param eps: Bias of power (default: ``1e-8``).
        :type eps: float

        """
        if self.wrapped is not NULL:
            raise TypeError("__init__() has already been called.")
        self.wrapped = new CppAdaGrad(eta, eps)

    def eta(self):
        """Returns the learning rate.

        :return: Learning rate.
        :rtype: float

        """
        return (<CppAdaGrad*> self.wrapped).eta()

    def eps(self):
        """Returns the hyperparameter eps.

        :return: The value of eps.
        :rtype: float

        """
        return (<CppAdaGrad*> self.wrapped).eps()


cdef class RMSProp(Optimizer):
    """RMSProp Optimizer.

    """

    def __init__(self, float eta = 0.01, float alpha = 0.9, float eps = 1e-8):
        """Creates a new RMSProp object.

        :param eta: Learning rate (default: ``0.01``).
        :type eta: float
        :param alpha: Decay factor of moment (default: ``0.9``).
        :type alpha: float
        :param eps: Bias of power (default: ``1e-8``).
        :type eps: float

        """
        if self.wrapped is not NULL:
            raise TypeError("__init__() has already been called.")
        self.wrapped = new CppRMSProp(eta, alpha, eps)

    def eta(self):
        """Returns the learning rate.

        :return: Learning rate.
        :rtype: float

        """
        return (<CppRMSProp*> self.wrapped).eta()

    def alpha(self):
        """Returns the hyperparameter alpha.

        :return: The value of alpha.
        :rtype: float

        """
        return (<CppRMSProp*> self.wrapped).alpha()

    def eps(self):
        """Returns the hyperparameter eps.

        :return: The value of eps.
        :rtype: float

        """
        return (<CppRMSProp*> self.wrapped).eps()


cdef class AdaDelta(Optimizer):
    """AdaDelta optimizer.
    https://arxiv.org/abs/1212.5701

    """

    def __init__(self, float rho = 0.95, float eps = 1e-6):
        """Creates a new AdaDelta object.

        :param rho: Decay factor of RMS operation (default: ``0.95``).
        :type rho: float
        :param eps: Bias of RMS values (default: ``1e-6``).
        :type eps: float

        """
        if self.wrapped is not NULL:
            raise TypeError("__init__() has already been called.")
        self.wrapped = new CppAdaDelta(rho, eps)

    def rho(self):
        """Returns the hyperparameter rho.

        :return: The value of rho.
        :rtype: float

        """
        return (<CppAdaDelta*> self.wrapped).rho()

    def eps(self):
        """Returns the hyperparameter eps.

        :return: The value of eps.
        :rtype: float

        """
        return (<CppAdaDelta*> self.wrapped).eps()


cdef class Adam(Optimizer):
    """Adam optimizer.
    https://arxiv.org/abs/1412.6980

    """

    def __init__(self, float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8):
        """Creates a new Adam object.

        :param alpha: Learning rate (default: ``0.001``).
        :type alpha: float
        :param beta1: Decay factor of momentum history (default: ``0.9``).
        :type beta1: float
        :param beta2: Decay factor of power history (default: ``0.999``).
        :type beta2: float
        :param eps: Bias of power (default: ``1e-8``).
        :type eps: float

        """
        if self.wrapped is not NULL:
            raise TypeError("__init__() has already been called.")
        self.wrapped = new CppAdam(alpha, beta1, beta2, eps)

    def alpha(self):
        """Returns the hyperparameter alpha.

        :return: The value of alpha.
        :rtype: float

        """
        return (<CppAdam*> self.wrapped).alpha()

    def beta1(self):
        """Returns the hyperparameter beta1.

        :return: The value of beta1.
        :rtype: float

        """
        return (<CppAdam*> self.wrapped).beta1()

    def beta2(self):
        """Returns the hyperparameter beta2.

        :return: The value of beta2.
        :rtype: float

        """
        return (<CppAdam*> self.wrapped).beta2()

    def eps(self):
        """Returns the hyperparameter eps.

        :return: The value of eps.
        :rtype: float

        """
        return (<CppAdam*> self.wrapped).eps()
