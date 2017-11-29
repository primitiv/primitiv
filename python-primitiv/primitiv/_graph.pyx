from libc.stdint cimport uintptr_t
from libcpp.vector cimport vector

from primitiv._device cimport Device
from primitiv._shape cimport wrapShape
from primitiv._tensor cimport Tensor
from primitiv._operator cimport op_pow, op_ipow, op_matmul
from primitiv.config cimport pystr_to_cppstr, cppstr_to_pystr

from weakref import WeakValueDictionary

cimport numpy as np
import numpy as np


# NOTE(vbkaisetsu):
# This is used for holding python instances related to C++.
# Without this variable, python instances are always created when C++ class
# instances are returned from functions.
# It means that users can not compare instances by using "is" operator.
cdef object py_primitiv_graph_weak_dict = WeakValueDictionary()


cdef class Node:
    """Pointer of a node in the computation graph.

    """

    def __init__(self, Node src = None):
        if src is None:
            self.wrapped = CppNode()
        else:
            self.wrapped = CppNode(src.wrapped)

    def valid(self):
        """Returns whether the node is valid or not.

        :return: ``True`` or ``False`` w.r.t. the node is valid or not.
        :rtype: bool

        """
        return self.wrapped.valid()

    def graph(self):
        """Returns corresponding Graph object.

        :return: Corresponding Graph object.
        :rtype: primitiv.Graph

        """
        return Graph.get_wrapper(&self.wrapped.graph())

    def function_id(self):
        """Returns the function ID.

        :return: Function ID.
        :rtype: int

        """
        return self.wrapped.function_id()

    def value_id(self):
        """Returns the value ID of the function.

        :return: Value ID.
        :rtype: int

        """
        return self.wrapped.value_id()

    def shape(self):
        """Returns shape of the node.

        :return: A Shape of this node.
        :rtype: primitiv.Shape

        """
        return wrapShape(self.wrapped.shape())

    def device(self):
        """Returns device of the node.

        :return: A Device of this node.
        :rtype: primitiv.Device

        """
        return Device.get_wrapper(&self.wrapped.device())

    def to_float(self):
        """Calculates the value of this node and returns a ``float``.

        :return: A calculated float value.
        :rtype: float

        This function calls ``Graph::forward()`` internally.
        This function can be used only when the Node has a scalar and
        non-minibatched shape (i.e., ``shape() == Shape()``)

        """
        cdef float val
        with nogil:
            val = self.wrapped.to_float()
        return val

    def to_list(self):
        """Calculates the value of this node and returns a list of float.

        :return: A list of calculated values.
        :rtype: list[float]

        This function calls ``Graph::forward()`` internally.

        """
        cdef vector[float] vec
        with nogil:
            vec = self.wrapped.to_vector()
        return vec

    def to_ndarrays(self):
        """Calculates the value of this node and returns a list of ``numpy.ndarray``
        containing ``numpy.float32``.

        :return: ``numpy.ndarray``'s list of calculated values.
        :rtype: list[numpy.ndarray[numpy.float32]]

        This function calls ``Graph::forward()`` internally.

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
        """Returns argmax indices along an axis of this node.

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
        """Returns argmin indices along an axis of this node.

        :param dim: A specified axis.
        :type dim: int
        :return: A list of integer that indicates positions of the minimum values.
        :rtype: list[int]

        """
        cdef vector[unsigned] vec
        with nogil:
            vec = self.wrapped.argmin(dim)
        return vec

    def backward(self):
        """Executes the backward operation from this node.

        """
        with nogil:
            self.wrapped.backward()

    def __pos__(self):
        return wrapNode(op_node_pos(self.wrapped))

    def __neg__(self):
        return wrapNode(op_node_neg(self.wrapped))

    def __add__(left, right):
        if isinstance(right, (int, float)):
            return wrapNode(op_node_add((<Node> left).wrapped, <float> right))
        elif isinstance(left, (int, float)):
            return wrapNode(op_node_add(<float> left, (<Node> right).wrapped))
        elif isinstance(left, Node) and isinstance(right, Node):
            return wrapNode(op_node_add((<Node> left).wrapped, (<Node> right).wrapped))
        else:
            return NotImplemented

    def __sub__(left, right):
        if isinstance(right, (int, float)):
            return wrapNode(op_node_sub((<Node> left).wrapped, <float> right))
        elif isinstance(left, (int, float)):
            return wrapNode(op_node_sub(<float> left, (<Node> right).wrapped))
        elif isinstance(left, Node) and isinstance(right, Node):
            return wrapNode(op_node_sub((<Node> left).wrapped, (<Node> right).wrapped))
        else:
            return NotImplemented

    def __mul__(left, right):
        if isinstance(right, (int, float)):
            return wrapNode(op_node_mul((<Node> left).wrapped, <float> right))
        elif isinstance(left, (int, float)):
            return wrapNode(op_node_mul(<float> left, (<Node> right).wrapped))
        elif isinstance(left, Node) and isinstance(right, Node):
            return wrapNode(op_node_mul((<Node> left).wrapped, (<Node> right).wrapped))
        else:
            return NotImplemented

    def __matmul__(left, right):
        if isinstance(left, Node) and isinstance(right, Node):
            return wrapNode(op_matmul((<Node> left).wrapped, (<Node> right).wrapped))
        else:
            return NotImplemented

    def __truediv__(left, right):
        if isinstance(right, (int, float)):
            return wrapNode(op_node_div((<Node> left).wrapped, <float> right))
        elif isinstance(left, (int, float)):
            return wrapNode(op_node_div(<float> left, (<Node> right).wrapped))
        elif isinstance(left, Node) and isinstance(right, Node):
            return wrapNode(op_node_div((<Node> left).wrapped, (<Node> right).wrapped))
        else:
            return NotImplemented

    def __pow__(left, right, mod):
        if mod is not None:
            return NotImplemented
        if isinstance(right, int) and -0x80000000 <= right <= 0x7fffffff:
            return wrapNode(op_ipow((<Node> left).wrapped, <int> right))
        elif isinstance(right, (int, float)):
            return wrapNode(op_pow((<Node> left).wrapped, <float> right))
        elif isinstance(left, (int, float)):
            return wrapNode(op_pow(<float> left, (<Node> right).wrapped))
        elif isinstance(left, Node) and isinstance(right, Node):
            return wrapNode(op_pow((<Node> left).wrapped, (<Node> right).wrapped))
        else:
            return NotImplemented

    def __copy__(self):
        raise NotImplementedError(type(self).__name__ + " does not support `__copy__` for now.")

    def __deepcopy__(self, memo):
        raise NotImplementedError(type(self).__name__ + " does not support `__deepcopy__` for now.")


cdef class Graph:
    """Computation graph.

    """

    def __init__(self):
        """Creates a new Graph object.

        """
        if self.wrapped is not NULL:
            raise TypeError("__init__() has already been called.")
        self.wrapped = new CppGraph()
        Graph.register_wrapper(self.wrapped, self)

    def __dealloc__(self):
        if self.wrapped is not NULL:
            del self.wrapped
            self.wrapped = NULL

    @staticmethod
    def get_default():
        """Retrieves the current default graph.

        :return: Reference of the current default graph.
        :rtype: primitiv.Graph
        :raises RuntimeError: if the default graph is null.

        """
        return Graph.get_wrapper(&CppGraph.get_default())

    @staticmethod
    def set_default(Graph g):
        """Specifies a new default graph.

        :param g: Reference of the new default graph.
        :type g: primitiv.Graph

        """
        CppGraph.set_default(g.wrapped[0])

    def clear(self):
        """Clear all functions in the graph.

        After calling this method, all Node objects supplied by the graph
        itself is invalidated.

        """
        self.wrapped.clear()
        return

    def forward(self, Node node):
        """Calculates the value of given node.

        :param node: Node object specifying the target node.
        :type node: primitiv.Node
        :return: Calculated value.
        :rtype: primitiv.Tensor

        This function calculates only the subgraph which is required to
        calculate the target node. Each intermediate result is stored to
        the corresponding node in the subgraph and they are re-used for
        future calculation. I.e., each node is calculated only once while
        the lifetime of the Graph object.

        """
        cdef CppTensor t
        with nogil:
            t = self.wrapped.forward(node.wrapped)
        return Tensor.get_wrapper_with_new(new CppTensor(t))

    def backward(self, Node node):
        """Calculates the backpropagation.

        :param node: Node object specifying the output node.
        :type node: primitiv.Node

        If ``node`` is not yet forwarded, this function implicitly calls
        ``forward(node)``.

        """
        with nogil:
            self.wrapped.backward(node.wrapped)
        return

    def get_shape(self, Node node):
        """Retrieves the shape of the node.

        :param node: Node object specifying the target node.
        :type node: primitiv.Node
        :return: The shape of the node.
        :rtype: primitiv.Shape

        """
        return wrapShape(self.wrapped.get_shape(node.wrapped))

    def get_device(self, Node node):
        """Retrieves the device of the node.

        :param node: Node object specifying the target node.
        :type node: primitiv.Node
        :return: Device of the node.
        :rtype: primitiv.Device

        """
        return Device.get_wrapper(&self.wrapped.get_device(node.wrapped))

    def dump(self, str fmt):
        """Dump internal graph structure.

        :param fmt: Name of the format.
        :type fmt: str
        :return: A string that represents the internal graph using given format.
        :rtype: str

        Available formats:

        * ``dot``: Graphviz's dot format.

        """
        return cppstr_to_pystr(self.wrapped.dump(pystr_to_cppstr(fmt)))

    def num_functions(self):
        """Returns the number of functions in the computation graph.

        :return: Number of nodes.
        :rtype: int

        """
        return self.wrapped.num_functions()

    def __copy__(self):
        raise NotImplementedError(type(self).__name__ + " does not support `__copy__` for now.")

    def __deepcopy__(self, memo):
        raise NotImplementedError(type(self).__name__ + " does not support `__deepcopy__` for now.")

    @staticmethod
    cdef void register_wrapper(CppGraph *ptr, Graph wrapper):
        if <uintptr_t> ptr in py_primitiv_graph_weak_dict:
            raise ValueError("Attempted to register the same C++ object twice.")
        py_primitiv_graph_weak_dict[<uintptr_t> ptr] = wrapper

    @staticmethod
    cdef Graph get_wrapper(CppGraph *ptr):
        # NOTE(vbkaisetsu):
        # Graph instances should be created and be registered before this
        # function is called.
        return py_primitiv_graph_weak_dict[<uintptr_t> ptr]
