from libc.stdint cimport uintptr_t
from libcpp.vector cimport vector

from primitiv._device cimport _Device
from primitiv._shape cimport wrapShape
from primitiv._tensor cimport _Tensor
from primitiv._operator cimport op_pow, op_ipow, op_matmul

from weakref import WeakValueDictionary

cimport numpy as np
import numpy as np


# NOTE(vbkaisetsu):
# This is used for holding python instances related to C++.
# Without this variable, python instances are always created when C++ class
# instances are returned from functions.
# It means that users can not compare instances by using "is" operator.
cdef object py_primitiv_graph_weak_dict = WeakValueDictionary()


cdef class _Node:

    def __init__(self, _Node src = None):
        if src is None:
            self.wrapped = CppNode()
        else:
            self.wrapped = CppNode(src.wrapped)

    def valid(self):
        return self.wrapped.valid()

    def graph(self):
        return _Graph.get_wrapper(&self.wrapped.graph())

    def function_id(self):
        return self.wrapped.function_id()

    def value_id(self):
        return self.wrapped.value_id()

    def shape(self):
        return wrapShape(self.wrapped.shape())

    def device(self):
        return _Device.get_wrapper(&self.wrapped.device())

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

    def backward(self):
        with nogil:
            self.wrapped.backward()

    def __pos__(self):
        return wrapNode(op_node_pos(self.wrapped))

    def __neg__(self):
        return wrapNode(op_node_neg(self.wrapped))

    def __add__(left, right):
        if isinstance(right, (int, float)):
            return wrapNode(op_node_add((<_Node> left).wrapped, <float> right))
        elif isinstance(left, (int, float)):
            return wrapNode(op_node_add(<float> left, (<_Node> right).wrapped))
        elif isinstance(left, _Node) and isinstance(right, _Node):
            return wrapNode(op_node_add((<_Node> left).wrapped, (<_Node> right).wrapped))
        else:
            return NotImplemented

    def __sub__(left, right):
        if isinstance(right, (int, float)):
            return wrapNode(op_node_sub((<_Node> left).wrapped, <float> right))
        elif isinstance(left, (int, float)):
            return wrapNode(op_node_sub(<float> left, (<_Node> right).wrapped))
        elif isinstance(left, _Node) and isinstance(right, _Node):
            return wrapNode(op_node_sub((<_Node> left).wrapped, (<_Node> right).wrapped))
        else:
            return NotImplemented

    def __mul__(left, right):
        if isinstance(right, (int, float)):
            return wrapNode(op_node_mul((<_Node> left).wrapped, <float> right))
        elif isinstance(left, (int, float)):
            return wrapNode(op_node_mul(<float> left, (<_Node> right).wrapped))
        elif isinstance(left, _Node) and isinstance(right, _Node):
            return wrapNode(op_node_mul((<_Node> left).wrapped, (<_Node> right).wrapped))
        else:
            return NotImplemented

    def __matmul__(left, right):
        if isinstance(left, _Node) and isinstance(right, _Node):
            return wrapNode(op_matmul((<_Node> left).wrapped, (<_Node> right).wrapped))
        else:
            return NotImplemented

    def __truediv__(left, right):
        if isinstance(right, (int, float)):
            return wrapNode(op_node_div((<_Node> left).wrapped, <float> right))
        elif isinstance(left, (int, float)):
            return wrapNode(op_node_div(<float> left, (<_Node> right).wrapped))
        elif isinstance(left, _Node) and isinstance(right, _Node):
            return wrapNode(op_node_div((<_Node> left).wrapped, (<_Node> right).wrapped))
        else:
            return NotImplemented

    def __pow__(left, right, mod):
        if mod is not None:
            return NotImplemented
        if isinstance(right, int) and -0x80000000 <= right <= 0x7fffffff:
            return wrapNode(op_ipow((<_Node> left).wrapped, <int> right))
        elif isinstance(right, (int, float)):
            return wrapNode(op_pow((<_Node> left).wrapped, <float> right))
        elif isinstance(left, (int, float)):
            return wrapNode(op_pow(<float> left, (<_Node> right).wrapped))
        elif isinstance(left, _Node) and isinstance(right, _Node):
            return wrapNode(op_pow((<_Node> left).wrapped, (<_Node> right).wrapped))
        else:
            return NotImplemented

    def __copy__(self):
        raise NotImplementedError(type(self).__name__ + " does not support `__copy__` for now.")

    def __deepcopy__(self, memo):
        raise NotImplementedError(type(self).__name__ + " does not support `__deepcopy__` for now.")


cdef class _Graph:

    def __init__(self):
        if self.wrapped is not NULL:
            raise MemoryError()
        self.wrapped = new CppGraph()
        _Graph.register_wrapper(self.wrapped, self)

    def __dealloc__(self):
        if self.wrapped is not NULL:
            del self.wrapped
            self.wrapped = NULL

    @staticmethod
    def get_default():
        return _Graph.get_wrapper(&CppGraph_get_default())

    @staticmethod
    def set_default(_Graph g):
        CppGraph_set_default(g.wrapped[0])

    def clear(self):
        self.wrapped.clear()
        return

    def forward(self, _Node node):
        cdef CppTensor t
        with nogil:
            t = self.wrapped.forward(node.wrapped)
        return _Tensor.get_wrapper_with_new(new CppTensor(t))

    def backward(self, _Node node):
        with nogil:
            self.wrapped.backward(node.wrapped)
        return

    def get_shape(self, _Node node):
        return wrapShape(self.wrapped.get_shape(node.wrapped))

    def get_device(self, _Node node):
        return _Device.get_wrapper(&self.wrapped.get_device(node.wrapped))

    def dump(self, str fmt):
        return self.wrapped.dump(<string> fmt.encode("utf-8")).decode("utf-8")

    def num_functions(self):
        return self.wrapped.num_functions()

    def __copy__(self):
        raise NotImplementedError(type(self).__name__ + " does not support `__copy__` for now.")

    def __deepcopy__(self, memo):
        raise NotImplementedError(type(self).__name__ + " does not support `__deepcopy__` for now.")

    @staticmethod
    cdef void register_wrapper(CppGraph *ptr, _Graph wrapper):
        if <uintptr_t> ptr in py_primitiv_graph_weak_dict:
            raise ValueError("Attempted to register the same C++ object twice.")
        py_primitiv_graph_weak_dict[<uintptr_t> ptr] = wrapper

    @staticmethod
    cdef _Graph get_wrapper(CppGraph *ptr):
        # NOTE(vbkaisetsu):
        # _Graph instances should be created and be registered before this
        # function is called.
        return py_primitiv_graph_weak_dict[<uintptr_t> ptr]
