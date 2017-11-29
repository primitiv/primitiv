from primitiv._tensor cimport Tensor


cdef class Initializer:
    """Abstract class to provide parameter initialization algorithms.

    """

    def apply(self, Tensor x):
        """Provides an initialized tensor.

        :param x: Tensor object to be initialized.
        :type x: primitiv.Tensor

        """
        self.wrapped.apply(x.wrapped[0])
        return

    def __copy__(self):
        raise NotImplementedError(type(self).__name__ + " does not support `__copy__` for now.")

    def __deepcopy__(self, memo):
        raise NotImplementedError(type(self).__name__ + " does not support `__deepcopy__` for now.")
