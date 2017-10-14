from libcpp.vector cimport vector

from primitiv._device cimport wrapDevice
from primitiv._shape cimport wrapShape
from primitiv._tensor cimport wrapTensor
from primitiv._operator cimport Node_pow, Node_matmul
from primitiv._graph cimport get_default as Graph_get_default
from primitiv._graph cimport set_default as Graph_set_default
from weakref import WeakValueDictionary

cimport numpy as np
import numpy as np


cdef class _Node:

    def __init__(self, _Node node = None):
        if node is None:
            self.wrapped = CppNode(node.wrapped)
        else:
            self.wrapped = CppNode()

    def valid(self):
        return self.wrapped.valid()

    def graph(self):
        return wrapGraph(&self.wrapped.graph())

    def function_id(self):
        return self.wrapped.function_id()

    def value_id(self):
        return self.wrapped.value_id()

    def shape(self):
        return wrapShape(self.wrapped.shape())

    def device(self):
        return wrapDevice(&self.wrapped.device())

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
            return wrapNode(Node_matmul((<_Node> left).wrapped, (<_Node> right).wrapped))
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
        if isinstance(right, (int, float)):
            return wrapNode(Node_pow((<_Node> left).wrapped, <float> right))
        elif isinstance(left, (int, float)):
            return wrapNode(Node_pow(<float> left, (<_Node> right).wrapped))
        elif isinstance(left, _Node) and isinstance(right, _Node):
            return wrapNode(Node_pow((<_Node> left).wrapped, (<_Node> right).wrapped))
        else:
            return NotImplemented


cdef class _Graph:

    def __init__(self):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        self.wrapped_newed = new CppGraph()
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

        global py_primitiv_graph_weak_dict
        if py_primitiv_graph_weak_dict is None:
            py_primitiv_graph_weak_dict = WeakValueDictionary()
        py_primitiv_graph_weak_dict[<uintptr_t> self.wrapped_newed] = self

    def __dealloc__(self):
        if self.wrapped_newed is not NULL:
            del self.wrapped_newed
            self.wrapped_newed = NULL

    @staticmethod
    def get_default():
        return wrapGraph(&Graph_get_default())

    @staticmethod
    def set_default(g):
        Graph_set_default((<_Graph> g).wrapped[0])

    def clear(self):
        self.wrapped.clear()
        return

    def forward(self, _Node node):
        cdef CppTensor t
        with nogil:
            t = self.wrapped.forward(node.wrapped)
        return wrapTensor(t)

    def backward(self, _Node node):
        with nogil:
            self.wrapped.backward(node.wrapped)
        return

    def get_shape(self, _Node node):
        return wrapShape(self.wrapped.get_shape(node.wrapped))

    def get_device(self, _Node node):
        return wrapDevice(&self.wrapped.get_device(node.wrapped))

    def dump(self):
        self.wrapped.dump()
        return

    def num_functions(self):
        return self.wrapped.num_functions()
