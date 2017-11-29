from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from primitiv._device cimport CppDevice
from primitiv._shape cimport CppShape
from primitiv._tensor cimport CppTensor


cdef extern from "primitiv/graph.h" nogil:
    cdef cppclass CppNode "primitiv::Node":
        CppNode(CppNode &&src) except +
        CppNode() except +
        bool valid() except +
        CppGraph &graph() except +
        unsigned function_id() except +
        unsigned value_id() except +
        const CppShape &shape() except +
        CppDevice &device() except +
        float to_float() except +
        vector[float] to_vector() except +
        vector[unsigned] argmax(unsigned dim) except +
        vector[unsigned] argmin(unsigned dim) except +
        void backward() except +


cdef extern from "node_op.h" namespace "python_primitiv_node":
    cdef CppNode op_node_pos(const CppNode &x) except +
    cdef CppNode op_node_neg(const CppNode &x) except +
    cdef CppNode op_node_add(const CppNode &x, float k) except +
    cdef CppNode op_node_add(float k, const CppNode &x) except +
    cdef CppNode op_node_add(const CppNode &a, const CppNode &b) except +
    cdef CppNode op_node_sub(const CppNode &x, float k) except +
    cdef CppNode op_node_sub(float k, const CppNode &x) except +
    cdef CppNode op_node_sub(const CppNode &a, const CppNode &b) except +
    cdef CppNode op_node_mul(const CppNode &x, float k) except +
    cdef CppNode op_node_mul(float k, const CppNode &x) except +
    cdef CppNode op_node_mul(const CppNode &a, const CppNode &b) except +
    cdef CppNode op_node_div(const CppNode &x, float k) except +
    cdef CppNode op_node_div(float k, const CppNode &x) except +
    cdef CppNode op_node_div(const CppNode &a, const CppNode &b) except +


cdef extern from "primitiv/graph.h" nogil:
    cdef cppclass CppGraph "primitiv::Graph":
        CppGraph() except +
        @staticmethod
        CppGraph &get_default() except +
        @staticmethod
        void set_default(CppGraph &g) except +
        void clear() except +
        const CppTensor &forward(const CppNode &node) except +
        void backward(const CppNode &node) except +
        const CppShape &get_shape(const CppNode &node) except +
        CppDevice &get_device(const CppNode &node) except +
        string dump(const string &format) except +
        unsigned num_functions() except +


cdef class Node:
    cdef CppNode wrapped


cdef class Graph:
    cdef CppGraph *wrapped
    cdef object __weakref__
    @staticmethod
    cdef void register_wrapper(CppGraph *ptr, Graph wrapper)
    @staticmethod
    cdef Graph get_wrapper(CppGraph *ptr)


cdef inline Node wrapNode(CppNode wrapped):
    cdef Node node = Node.__new__(Node)
    node.wrapped = wrapped
    return node
