cdef class Constant(Initializer):
    """Initializer to generate a same-value tensor.

    """

    def __init__(self, float k):
        """Crates a new initializer object.

        :param k: Constant to provide.
        :type k: float

        """
        if self.wrapped_newed is not NULL:
            raise TypeError("__init__() has already been called.")
        self.wrapped_newed = new CppConstant(k)
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef CppConstant *temp
        if self.wrapped_newed is not NULL:
            temp = <CppConstant*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL


cdef class Uniform(Initializer):
    """Initializer using a parameterized uniform distribution (lower, upper].

    """

    def __init__(self, float lower, float upper):
        """Crates a new initializer object.

        :param lower: Lower bound of the distribusion.
        :type lower: float
        :param upper: Upper bound of the distribusion.
        :type upper: float

        """
        if self.wrapped_newed is not NULL:
            raise TypeError("__init__() has already been called.")
        self.wrapped_newed = new CppUniform(lower, upper)
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef CppUniform *temp
        if self.wrapped_newed is not NULL:
            temp = <CppUniform*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL


cdef class Normal(Initializer):
    """Initializer using a parameterized normal distribution N(mean, sd).

    """

    def __init__(self, float mean, float sd):
        """Crates a new initializer object.

        :param mean: Mean value of the distribusion.
        :type mean: float
        :param sd: Standard deviation of the distribusion.
        :type sd: float

        """
        if self.wrapped_newed is not NULL:
            raise TypeError("__init__() has already been called.")
        self.wrapped_newed = new CppNormal(mean, sd)
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef CppNormal *temp
        if self.wrapped_newed is not NULL:
            temp = <CppNormal*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL


cdef class Identity(Initializer):
    """Identity matrix initializer.

    """

    def __init__(self):
        """Crates a new initializer object.

        """
        if self.wrapped_newed is not NULL:
            raise TypeError("__init__() has already been called.")
        self.wrapped_newed = new CppIdentity()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef CppIdentity *temp
        if self.wrapped_newed is not NULL:
            temp = <CppIdentity*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL


cdef class XavierUniform(Initializer):
    """The Xavier matrix initialization with the uniform distribution.

    """

    def __init__(self, scale = 1.0):
        """Crates a new initializer object.

        :param scale: Scale of the distribusion.
        :type scale: float

        """
        if self.wrapped_newed is not NULL:
            raise TypeError("__init__() has already been called.")
        self.wrapped_newed = new CppXavierUniform(scale)
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef CppXavierUniform *temp
        if self.wrapped_newed is not NULL:
            temp = <CppXavierUniform*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL


cdef class XavierNormal(Initializer):
    """The Xavier matrix initialization with the normal distribution.

    """

    def __init__(self, float scale = 1.0):
        """Crates a new initializer object.

        :param scale: Scale of the distribusion.
        :type scale: float

        """
        if self.wrapped_newed is not NULL:
            raise TypeError("__init__() has already been called.")
        self.wrapped_newed = new CppXavierNormal(scale)
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef CppXavierNormal *temp
        if self.wrapped_newed is not NULL:
            temp = <CppXavierNormal*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL
