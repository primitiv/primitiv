from libcpp.vector cimport vector

from primitiv._device cimport wrapDevice
from primitiv._shape cimport _Shape, wrapShape, normShape

cimport numpy as np
import numpy as np


cdef class _Tensor:

    def __init__(self, _Tensor src = None):
        if src is None:
            self.wrapped = CppTensor()
        else:
            self.wrapped = CppTensor(src.wrapped)
        return

    def valid(self):
        return self.wrapped.valid()

    def shape(self):
        return wrapShape(self.wrapped.shape())

    def device(self):
        return wrapDevice(&self.wrapped.device())

    #def data(self):
        #return self.wrapped.data()

    def to_float(self):
        cdef float val
        with nogil:
            val = self.wrapped.to_float()
        return val

    def to_list(self):
        cdef vector[float] vec
        with nogil:
            vec = self.wrapped.to_vector()
        return vec

    def to_ndarrays(self):
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
        cdef vector[unsigned] vec
        with nogil:
            vec = self.wrapped.argmax(dim)
        return vec

    def argmin(self, unsigned dim):
        cdef vector[unsigned] vec
        with nogil:
            vec = self.wrapped.argmin(dim)
        return vec

    def reset(self, float k):
        self.wrapped.reset(k)

    #def reset_by_array(self, vector[float] values):
        #self.wrapped.reset_by_array(values)

    def reset_by_vector(self, vector[float] values):
        self.wrapped.reset_by_vector(values)

    def reshape(self, _Shape new_shape):
        self.wrapped = self.wrapped.reshape(normShape(new_shape).wrapped)
        return self

    def flatten(self):
        return wrapTensor(self.wrapped.flatten())

    def __imul__(self, float k):
        self.wrapped = tensor_inplace_multiply_const(self.wrapped, k)
        return self

    def __iadd__(self, _Tensor x):
        self.wrapped = tensor_inplace_add(self.wrapped, x.wrapped)
        return self

    def __isub__(self, _Tensor x):
        self.wrapped = tensor_inplace_subtract(self.wrapped, x.wrapped)
        return self

    def __copy__(self):
        raise NotImplementedError(type(self).__name__ + " does not support `__copy__` for now.")

    def __deepcopy__(self, memo):
        raise NotImplementedError(type(self).__name__ + " does not support `__deepcopy__` for now.")
