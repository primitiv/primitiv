from libcpp.vector cimport vector

from primitiv.device cimport wrapDevice
from primitiv.shape cimport wrapShape
from primitiv.tensor cimport wrapTensor

cimport numpy as np
import numpy as np


cdef class _Node:

    def __init__(self, _Node node = None):
        if node is None:
            self.wrapped = Node(node.wrapped)
        else:
            self.wrapped = Node()

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
        return self.wrapped.to_vector()

    def to_ndarrays(self):
        cdef vector[float] vec = self.wrapped.to_vector()
        cdef Shape s = self.wrapped.shape()
        cdef np.ndarray output_item
        cdef np.float32_t *np_data
        cdef unsigned volume = s.volume()
        output = []
        for j in range(s.batch()):
            output_item = np.empty([s[i] for i in range(s.depth())], dtype=np.float32, order="F")
            np_data = <np.float32_t*> output_item.data
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

    def __truediv__(left, right):
        if isinstance(right, (int, float)):
            return wrapNode(op_node_div((<_Node> left).wrapped, <float> right))
        elif isinstance(left, (int, float)):
            return wrapNode(op_node_div(<float> left, (<_Node> right).wrapped))
        elif isinstance(left, _Node) and isinstance(right, _Node):
            return wrapNode(op_node_div((<_Node> left).wrapped, (<_Node> right).wrapped))
        else:
            return NotImplemented


cdef class _Graph:

    def __init__(self):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        self.wrapped_newed = new Graph()
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        if self.wrapped_newed is not NULL:
            del self.wrapped_newed
            self.wrapped_newed = NULL

    def forward(self, _Node node):
        return wrapTensor(self.wrapped.forward(node.wrapped))

    def backward(self, _Node node):
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
