from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libcpp cimport bool
from primitiv.device cimport Device
from primitiv.shape cimport Shape
from primitiv.tensor cimport Tensor


cdef extern from "primitiv/graph.h" namespace "primitiv":
    cdef cppclass Node:
        Node(Node &&src) except +
        Node() except +
        bool valid() except +
        Graph &graph() except +
        unsigned function_id() except +
        unsigned value_id() except +
        const Shape &shape() except +
        Device &device() except +
        vector[float] to_vector() except +


cdef extern from "node_op.h" namespace "python_primitiv_node":
    cdef Node op_node_pos(const Node &x) except +
    cdef Node op_node_neg(const Node &x) except +
    cdef Node op_node_add(const Node &x, float k) except +
    cdef Node op_node_add(float k, const Node &x) except +
    cdef Node op_node_add(const Node &a, const Node &b) except +
    cdef Node op_node_sub(const Node &x, float k) except +
    cdef Node op_node_sub(float k, const Node &x) except +
    cdef Node op_node_sub(const Node &a, const Node &b) except +
    cdef Node op_node_mul(const Node &x, float k) except +
    cdef Node op_node_mul(float k, const Node &x) except +
    cdef Node op_node_mul(const Node &a, const Node &b) except +
    cdef Node op_node_div(const Node &x, float k) except +
    cdef Node op_node_div(float k, const Node &x) except +
    cdef Node op_node_div(const Node &a, const Node &b) except +


cdef extern from "primitiv/graph.h" namespace "primitiv":
    cdef cppclass Graph:
        Graph() except +
        const Tensor &forward(const Node &node) except +
        void backward(const Node &node) except +
        const Shape &get_shape(const Node &node) except +
        Device &get_device(const Node &node) except +
        void dump() except +
        unsigned num_functions() except +


cdef class _Node:
    cdef Node wrapped


cdef class _Graph:
    cdef Graph *wrapped
    cdef _Graph with_graph_stack


cdef inline _Node wrapNode(Node wrapped) except +:
    cdef _Node node = _Node.__new__(_Node)
    node.wrapped = wrapped
    return node


cdef inline _Graph wrapGraph(Graph *wrapped) except +:
    cdef _Graph graph = _Graph.__new__(_Graph)
    graph.wrapped = wrapped
    return graph
