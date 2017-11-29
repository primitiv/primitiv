from libc.stdint cimport uintptr_t
from libcpp.vector cimport vector

from primitiv._device cimport Device
from primitiv._shape cimport Shape, wrapShape, normShape
from primitiv._operator cimport op_pow, op_ipow, op_matmul

from weakref import WeakValueDictionary

cimport numpy as np
import numpy as np

# NOTE(vbkaisetsu):
# This is used for holding python instances related to C++.
# Without this variable, python instances are always created when C++ class
# instances are returned from functions.
# It means that users can not compare instances by using "is" operator.
cdef object py_primitiv_tensor_weak_dict = WeakValueDictionary()


cdef class Tensor:
    """Value with any dimensions.

    """

    def __init__(self, Tensor src = None):
        if self.wrapped is not NULL:
            raise TypeError("__init__() has already been called.")
        if src is None:
            self.wrapped = new CppTensor()
        else:
            self.wrapped = new CppTensor(src.wrapped[0])
        self.del_required = True
        Tensor.register_wrapper(self.wrapped, self)

    def __dealloc__(self):
        if self.del_required:
            del self.wrapped
            self.wrapped = NULL

    def valid(self):
        """Check whether the object is valid or not.

        :return: ``True`` if the object is valid, ``False`` otherwise.
        :rtype: bool

        This returns ``False`` when the object is created through the default
        constructor or the object had been moved.

        """
        return self.wrapped.valid()

    def shape(self):
        """Returns the shape of the Tensor.

        :return: Shape of the Tensor.
        :rtype: primitiv.Shape

        """
        return wrapShape(self.wrapped.shape())

    def device(self):
        """Returns the Device object related to the internal memory.

        :return: Device object.
        :rtype: primitiv.Device

        """
        return Device.get_wrapper(&self.wrapped.device())

    #def data(self):
        #return self.wrapped.data()

    def to_float(self):
        """Retrieves one internal value in the tensor.

        :return: An internal float value.
        :rtype: float

        This function can be used only when the tensor is a scalar and
        non-minibatched (i.e., ``shape() == Shape()``).

        """
        cdef float val
        with nogil:
            val = self.wrapped.to_float()
        return val

    def to_list(self):
        """Retrieves internal values in the tensor as a ``list``.

        :return: A list of the internal values.
        :rtype: list[float]

        Each resulting values a reordered by the column-major order, and
        the batch size is assumed as the last dimension of the tensor.

        """
        cdef vector[float] vec
        with nogil:
            vec = self.wrapped.to_vector()
        return vec

    def to_ndarrays(self):
        """Retrieves internal values in the tensor as a  list of ``numpy.ndarray``
        containing ``numpy.float32``.

        :return: ``numpy.ndarray``'s list of the internal values.
        :rtype: list[numpy.ndarray[numpy.float32]]

        """
        cdef vector[float] vec
        cdef CppShape s = self.wrapped.shape()
        cdef np.ndarray output_item
        cdef np.float32_t *np_data
        cdef unsigned volume = s.volume()
        cdef unsigned j, i
        with nogil:
            vec = self.wrapped.to_vector()
        output = []
        for j in range(s.batch()):
            output_item = np.empty([s[i] for i in range(s.depth())], dtype=np.float32, order="F")
            np_data = <np.float32_t*> output_item.data
            with nogil:
                for i in range(volume):
                    np_data[i] = vec[i + j * volume]
            output.append(output_item)
        return output

    def argmax(self, unsigned dim):
        """Retrieves argmax indices along an axis.

        :param dim: A specified axis.
        :type dim: int
        :return: A list of integer that indicates positions of the maximum values.
        :rtype: list[int]

        """
        cdef vector[unsigned] vec
        with nogil:
            vec = self.wrapped.argmax(dim)
        return vec

    def argmin(self, unsigned dim):
        """Retrieves argmin indices along an axis.

        :param dim: A specified axis.
        :type dim: int
        :return: A list of integer that indicates positions of the minimum values.
        :rtype: list[int]

        """
        cdef vector[unsigned] vec
        with nogil:
            vec = self.wrapped.argmin(dim)
        return vec

    def reset(self, float k):
        """Reset internal values using a constant.

        :param k: A value to be used to initialize each element.
        :type k: float

        """
        self.wrapped.reset(k)

    #def reset_by_array(self, vector[float] values):
        #self.wrapped.reset_by_array(values)

    def reset_by_vector(self, vector[float] values):
        """Reset internal values using a list.

        :param values: list of values to be used to initialize each element.
        :type values: list[float]

        ``len(values)`` should be equal to ``shape().size()``. Each element
        should be ordered by the column-major order, and the batch size is
        assumed as the last dimension.

        """
        self.wrapped.reset_by_vector(values)

    def reshape(self, Shape new_shape):
        """Returns a tensor which have the same values and different shape.

        :param new_shape: New shape with batch size 1.
        :type new_shape: primitiv.Shape
        :return: A new tensor.
        :rtype: primitiv.Tensor

        """
        return Tensor.get_wrapper_with_new(new CppTensor(self.wrapped.reshape(normShape(new_shape).wrapped)))

    def flatten(self):
        """Returns a flattened tensor.

        :return: A new tensor.
        :rtype: primitiv.Tensor

        """
        return Tensor.get_wrapper_with_new(new CppTensor(self.wrapped.flatten()))

    def __pos__(self):
        return Tensor.get_wrapper_with_new(new CppTensor(op_tensor_pos(self.wrapped[0])))

    def __neg__(self):
        return Tensor.get_wrapper_with_new(new CppTensor(op_tensor_neg(self.wrapped[0])))

    def __add__(left, right):
        if isinstance(right, (int, float)):
            return Tensor.get_wrapper_with_new(new CppTensor(op_tensor_add((<Tensor> left).wrapped[0], <float> right)))
        elif isinstance(left, (int, float)):
            return Tensor.get_wrapper_with_new(new CppTensor(op_tensor_add(<float> left, (<Tensor> right).wrapped[0])))
        elif isinstance(left, Tensor) and isinstance(right, Tensor):
            return Tensor.get_wrapper_with_new(new CppTensor(op_tensor_add((<Tensor> left).wrapped[0], (<Tensor> right).wrapped[0])))
        else:
            return NotImplemented

    def __sub__(left, right):
        if isinstance(right, (int, float)):
            return Tensor.get_wrapper_with_new(new CppTensor(op_tensor_sub((<Tensor> left).wrapped[0], <float> right)))
        elif isinstance(left, (int, float)):
            return Tensor.get_wrapper_with_new(new CppTensor(op_tensor_sub(<float> left, (<Tensor> right).wrapped[0])))
        elif isinstance(left, Tensor) and isinstance(right, Tensor):
            return Tensor.get_wrapper_with_new(new CppTensor(op_tensor_sub((<Tensor> left).wrapped[0], (<Tensor> right).wrapped[0])))
        else:
            return NotImplemented

    def __mul__(left, right):
        if isinstance(right, (int, float)):
            return Tensor.get_wrapper_with_new(new CppTensor(op_tensor_mul((<Tensor> left).wrapped[0], <float> right)))
        elif isinstance(left, (int, float)):
            return Tensor.get_wrapper_with_new(new CppTensor(op_tensor_mul(<float> left, (<Tensor> right).wrapped[0])))
        elif isinstance(left, Tensor) and isinstance(right, Tensor):
            return Tensor.get_wrapper_with_new(new CppTensor(op_tensor_mul((<Tensor> left).wrapped[0], (<Tensor> right).wrapped[0])))
        else:
            return NotImplemented

    def __matmul__(left, right):
        if isinstance(left, Tensor) and isinstance(right, Tensor):
            return Tensor.get_wrapper_with_new(new CppTensor(op_matmul((<Tensor> left).wrapped[0], (<Tensor> right).wrapped[0])))
        else:
            return NotImplemented

    def __truediv__(left, right):
        if isinstance(right, (int, float)):
            return Tensor.get_wrapper_with_new(new CppTensor(op_tensor_div((<Tensor> left).wrapped[0], <float> right)))
        elif isinstance(left, (int, float)):
            return Tensor.get_wrapper_with_new(new CppTensor(op_tensor_div(<float> left, (<Tensor> right).wrapped[0])))
        elif isinstance(left, Tensor) and isinstance(right, Tensor):
            return Tensor.get_wrapper_with_new(new CppTensor(op_tensor_div((<Tensor> left).wrapped[0], (<Tensor> right).wrapped[0])))
        else:
            return NotImplemented

    def __pow__(left, right, mod):
        if mod is not None:
            return NotImplemented
        if isinstance(right, int) and -0x80000000 <= right <= 0x7fffffff:
            return Tensor.get_wrapper_with_new(new CppTensor(op_ipow((<Tensor> left).wrapped[0], <int> right)))
        elif isinstance(right, (int, float)):
            return Tensor.get_wrapper_with_new(new CppTensor(op_pow((<Tensor> left).wrapped[0], <float> right)))
        elif isinstance(left, (int, float)):
            return Tensor.get_wrapper_with_new(new CppTensor(op_pow(<float> left, (<Tensor> right).wrapped[0])))
        elif isinstance(left, Tensor) and isinstance(right, Tensor):
            return Tensor.get_wrapper_with_new(new CppTensor(op_pow((<Tensor> left).wrapped[0], (<Tensor> right).wrapped[0])))
        else:
            return NotImplemented

    def __imul__(self, float k):
        op_tensor_imul(self.wrapped[0], k)
        return self

    def __iadd__(self, Tensor x):
        op_tensor_iadd(self.wrapped[0], x.wrapped[0])
        return self

    def __isub__(self, Tensor x):
        op_tensor_isub(self.wrapped[0], x.wrapped[0])
        return self

    def __copy__(self):
        raise NotImplementedError(type(self).__name__ + " does not support `__copy__` for now.")

    def __deepcopy__(self, memo):
        raise NotImplementedError(type(self).__name__ + " does not support `__deepcopy__` for now.")

    @staticmethod
    cdef void register_wrapper(CppTensor *ptr, Tensor wrapper):
        if <uintptr_t> ptr in py_primitiv_tensor_weak_dict:
            raise ValueError("Attempted to register the same C++ object twice.")
        py_primitiv_tensor_weak_dict[<uintptr_t> ptr] = wrapper

    @staticmethod
    cdef Tensor get_wrapper(CppTensor *ptr):
        ret = py_primitiv_tensor_weak_dict.get(<uintptr_t> ptr)
        if ret:
            return ret
        cdef Tensor tensor = Tensor.__new__(Tensor)
        tensor.wrapped = ptr
        tensor.del_required = False
        py_primitiv_tensor_weak_dict[<uintptr_t> ptr] = tensor
        return tensor

    @staticmethod
    cdef Tensor get_wrapper_with_new(CppTensor *ptr):
        cdef Tensor tensor = Tensor.__new__(Tensor)
        tensor.wrapped = ptr
        tensor.del_required = False  # dummy to add this to dict.
        if py_primitiv_tensor_weak_dict.setdefault(<uintptr_t> ptr, tensor) is not tensor:
            raise ValueError("Attempted to register the same C++ object twice.")
        tensor.del_required = True
        return tensor
