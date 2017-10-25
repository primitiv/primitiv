from libcpp.vector cimport vector

from primitiv._device cimport wrapDevice
from primitiv._shape cimport _Shape, wrapShape, normShape
from primitiv._operator cimport op_pow, op_matmul

from weakref import WeakValueDictionary

cimport numpy as np
import numpy as np


cdef class _Tensor:

    def __init__(self, _Tensor src = None):
        if self.wrapped is not NULL:
            raise MemoryError()
        if src is None:
            self.wrapped = new CppTensor()
        else:
            self.wrapped = new CppTensor(src.wrapped[0])
        self.del_required = True

        global py_primitiv_tensor_weak_dict
        if py_primitiv_tensor_weak_dict is None:
            py_primitiv_tensor_weak_dict = WeakValueDictionary()
        py_primitiv_tensor_weak_dict[<uintptr_t> self.wrapped_newed] = self

    def __dealloc__(self):
        if self.del_required:
            del self.wrapped
            self.wrapped = NULL

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
        return wrapTensorWithNew(new CppTensor(self.wrapped.reshape(normShape(new_shape).wrapped)))

    def flatten(self):
        return wrapTensorWithNew(new CppTensor(self.wrapped.flatten()))

    def __pos__(self):
        return wrapTensorWithNew(new CppTensor(op_tensor_pos(self.wrapped[0])))

    def __neg__(self):
        return wrapTensorWithNew(new CppTensor(op_tensor_neg(self.wrapped[0])))

    def __add__(left, right):
        if isinstance(right, (int, float)):
            return wrapTensorWithNew(new CppTensor(op_tensor_add((<_Tensor> left).wrapped[0], <float> right)))
        elif isinstance(left, (int, float)):
            return wrapTensorWithNew(new CppTensor(op_tensor_add(<float> left, (<_Tensor> right).wrapped[0])))
        elif isinstance(left, _Tensor) and isinstance(right, _Tensor):
            return wrapTensorWithNew(new CppTensor(op_tensor_add((<_Tensor> left).wrapped[0], (<_Tensor> right).wrapped[0])))
        else:
            return NotImplemented

    def __sub__(left, right):
        if isinstance(right, (int, float)):
            return wrapTensorWithNew(new CppTensor(op_tensor_sub((<_Tensor> left).wrapped[0], <float> right)))
        elif isinstance(left, (int, float)):
            return wrapTensorWithNew(new CppTensor(op_tensor_sub(<float> left, (<_Tensor> right).wrapped[0])))
        elif isinstance(left, _Tensor) and isinstance(right, _Tensor):
            return wrapTensorWithNew(new CppTensor(op_tensor_sub((<_Tensor> left).wrapped[0], (<_Tensor> right).wrapped[0])))
        else:
            return NotImplemented

    def __mul__(left, right):
        if isinstance(right, (int, float)):
            return wrapTensorWithNew(new CppTensor(op_tensor_mul((<_Tensor> left).wrapped[0], <float> right)))
        elif isinstance(left, (int, float)):
            return wrapTensorWithNew(new CppTensor(op_tensor_mul(<float> left, (<_Tensor> right).wrapped[0])))
        elif isinstance(left, _Tensor) and isinstance(right, _Tensor):
            return wrapTensorWithNew(new CppTensor(op_tensor_mul((<_Tensor> left).wrapped[0], (<_Tensor> right).wrapped[0])))
        else:
            return NotImplemented

    def __matmul__(left, right):
        if isinstance(left, _Tensor) and isinstance(right, _Tensor):
            return wrapTensorWithNew(new CppTensor(op_matmul((<_Tensor> left).wrapped[0], (<_Tensor> right).wrapped[0])))
        else:
            return NotImplemented

    def __truediv__(left, right):
        if isinstance(right, (int, float)):
            return wrapTensorWithNew(new CppTensor(op_tensor_div((<_Tensor> left).wrapped[0], <float> right)))
        elif isinstance(left, (int, float)):
            return wrapTensorWithNew(new CppTensor(op_tensor_div(<float> left, (<_Tensor> right).wrapped[0])))
        elif isinstance(left, _Tensor) and isinstance(right, _Tensor):
            return wrapTensorWithNew(new CppTensor(op_tensor_div((<_Tensor> left).wrapped[0], (<_Tensor> right).wrapped[0])))
        else:
            return NotImplemented

    def __pow__(left, right, mod):
        if mod is not None:
            return NotImplemented
        if isinstance(right, (int, float)):
            return wrapTensorWithNew(new CppTensor(op_pow((<_Tensor> left).wrapped[0], <float> right)))
        elif isinstance(left, (int, float)):
            return wrapTensorWithNew(new CppTensor(op_pow(<float> left, (<_Tensor> right).wrapped[0])))
        elif isinstance(left, _Tensor) and isinstance(right, _Tensor):
            return wrapTensorWithNew(new CppTensor(op_pow((<_Tensor> left).wrapped[0], (<_Tensor> right).wrapped[0])))
        else:
            return NotImplemented

    def __imul__(self, float k):
        op_tensor_imul(self.wrapped[0], k)
        return self

    def __iadd__(self, _Tensor x):
        op_tensor_iadd(self.wrapped[0], x.wrapped[0])
        return self

    def __isub__(self, _Tensor x):
        op_tensor_isub(self.wrapped[0], x.wrapped[0])
        return self

    def __copy__(self):
        raise NotImplementedError(type(self).__name__ + " does not support `__copy__` for now.")

    def __deepcopy__(self, memo):
        raise NotImplementedError(type(self).__name__ + " does not support `__deepcopy__` for now.")
