from libcpp.vector cimport vector
from primitiv.config cimport cppstr_to_pystr


cdef class Shape:
    """Data structure to represent the shape of the node.

    """

    def __init__(self, dims = None, unsigned batch = 1):
        """Creates a new Shape object.

        :param dims: List of dimension size
        :type dims: list[int]
        :param batch: Batch size (default: 1)
        :type batch: int

        Examples:

            >>> Shape()         == Shape([1, 1, 1, ...], 1) # scalar
            >>> Shape([])       == Shape([1, 1, 1, ...], 1) # scalar
            >>> Shape([n])      == Shape([n, 1, 1, ...], 1) # row vector
            >>> Shape([n, m])   == Shape([n, m, 1, ...], 1) # matrix
            >>> Shape([...], k) # k-parallelized data (mini-batch)

        """
        if dims is None:
            self.wrapped = CppShape()
        else:
            self.wrapped = CppShape(<vector[unsigned]> dims, <unsigned> batch)

    def dims(self):
        """Returns the dimension array.

        :return: Copy of the dimension array.
        :rtype: list[int]

        """
        return self.wrapped.dims()

    def depth(self):
        """Returns the depth (length of non-1 dimensions) of the shape.

        :return: The depth of the shape.
        :rtype: int

        """
        return self.wrapped.depth()

    def batch(self):
        """Returns the batch size.

        :return: Batch size.
        :rtype: int

        """
        return self.wrapped.batch()

    def volume(self):
        """Returns the number of elements in each sample.
        This value is equal to the product of all dimensions.

        :return: Number of elements.
        :rtype: int

        """
        return self.wrapped.volume()

    def lower_volume(self, unsigned dim):
        """Returns the number of elements in 1 to specified dim.

        :param dim: Upper bound of the dimension.
        :type dim: int
        :return: ``dims[0] * dims[1] * ... * dims[dim-1]``
        :rtype: int

        """
        return self.wrapped.lower_volume(dim)

    def size(self):
        """Returns the number of elements in all samples of the mini-batch.
        This value is equal to ``batch() * volume()``.

        :return: Number of elements.
        :rtype: int

        """
        return self.wrapped.size()

    def __str__(self):
        return cppstr_to_pystr(self.wrapped.to_string())

    def __repr__(self):
        return "Shape([%s], %d)" % (", ".join(str(self[i]) for i in range(self.depth())), self.batch())

    def __getitem__(self, unsigned i):
        return self.wrapped[i]

    def __eq__(Shape self, Shape rhs):
        return self.wrapped == rhs.wrapped

    def __ne__(Shape self, Shape rhs):
        return self.wrapped != rhs.wrapped

    def has_batch(self):
        """Checks whether the shape has minibatch or not.

        :return: ``True`` if the shape has minibatch, ``False`` otherwise.
        :rtype: bool

        """
        return self.wrapped.has_batch()

    def has_compatible_batch(self, Shape rhs):
        """Checks whether two batch size is compatible (broadcastable) or not.

        :param rhs: Shape object to compare.
        :type rhs: primitiv.Shape
        :return: ``True`` if both batch size is compatible, ``False`` otherwise.
        :rtype: bool

        """
        return self.wrapped.has_compatible_batch(rhs.wrapped)

    def is_scalar(self):
        """Checks whether the shape is a scalar or not.

        :return: ``True`` if the shape is a scalar, ``False`` otherwise.
        :rtype: bool

        """
        return self.wrapped.is_scalar()

    def is_row_vector(self):
        """Checks whether the shape is a row vector or not.

        :return: ``True`` if the shape is a row vector, ``False`` otherwise.
        :rtype: bool

        """
        return self.wrapped.is_row_vector()

    def is_matrix(self):
        """Checks whether the shape is a vector or a matrix, or not.

        :return: ``True`` if the shape is a vector or a matrix, ``False`` otherwise.
        :rtype: bool

        """
        return self.wrapped.is_matrix()

    def has_same_dims(self, Shape rhs):
        """Checks whether two shapes have completely same dimensions.

        :param rhs: Shape object to compare.
        :type rhs: primitiv.Shape
        :return: ``True`` if both shape have same dimensions, ``False`` otherwise.
        :rtype: bool

        """
        return self.wrapped.has_same_dims(rhs.wrapped)

    def has_same_loo_dims(self, Shape rhs, unsigned dim):
        """Checks whether two shapes have same dimensions without an axis.
        (LOO: leave one out)

        :param rhs: Shape object to compare.
        :type rhs: primitiv.Shape
        :param dim: Dimension to be ignored.
        :type dim: int
        :return: ``True`` if both shape have same dimensions regardless the dimension
          ``dim``, ``False`` otherwise.
        :rtype: bool

        """
        return self.wrapped.has_same_loo_dims(rhs.wrapped, dim)

    def resize_dim(self, unsigned dim, unsigned m):
        """Creates a new shape which have one different dimension.

        :param dim: Dimension to be changed.
        :type dim: int
        :param m: New size of the dimension ``dim``.
        :type m: int
        :return: New shape.
        :rtype: primitiv.Shape

        """
        self.wrapped.resize_dim(dim, m)
        return self

    def resize_batch(self, unsigned batch):
        """Creates a new shape which have specified batch size.

        :param batch: New batch size.
        :type batch: int
        :return: New shape.
        :rtype: primitiv.Shape

        """
        self.wrapped.resize_batch(batch)
        return self

    def update_dim(self, unsigned dim, unsigned m):
        """Directly updates a specified dimension.

        :param dim: Dimension to be updated.
        :type dim: int
        :param m: New size of the dimension ``dim``.
        :type m: int

        """
        self.wrapped.update_dim(dim, m)

    def update_batch(self, unsigned batch):
        """Directly updates the batch size.

        :param batch: New batch size.
        :type batch: int

        """
        self.wrapped.update_batch(batch)

    def __copy__(self):
        raise NotImplementedError(type(self).__name__ + " does not support `__copy__` for now.")

    def __deepcopy__(self, memo):
        raise NotImplementedError(type(self).__name__ + " does not support `__deepcopy__` for now.")
