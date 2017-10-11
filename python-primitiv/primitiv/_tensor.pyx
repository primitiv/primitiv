from libcpp.vector cimport vector

from primitiv._device cimport wrapDevice
from primitiv._shape cimport _Shape, wrapShape, normShape

cimport numpy as np
import numpy as np


cdef class _Tensor:

    def __init__(self, src = None):
        if src is None:
            self.wrapped = Tensor()
        else:
            self.wrapped = Tensor((<_Tensor> src).wrapped)
        return

    def valid(self):
        return self.wrapped.valid()

    def shape(self):
        return wrapShape(self.wrapped.shape())

    def device(self):
        return wrapDevice(&self.wrapped.device())

    #def data(self):
        #return self.wrapped.data()

    def to_list(self):
        cdef vector[float] vec
        with nogil:
            vec = self.wrapped.to_vector()
        return vec

    def to_ndarrays(self):
        cdef vector[float] vec
        cdef Shape s = self.wrapped.shape()
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
