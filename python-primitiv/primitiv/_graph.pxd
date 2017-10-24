from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libcpp cimport bool
from primitiv._device cimport CppDevice
from primitiv._shape cimport CppShape
from primitiv._tensor cimport CppTensor
from libc.stdint cimport uintptr_t


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
        void clear() except +
        const CppTensor &forward(const CppNode &node) except +
        void backward(const CppNode &node) except +
        const CppShape &get_shape(const CppNode &node) except +
        CppDevice &get_device(const CppNode &node) except +
        string dump(const string &format) except +
        unsigned num_functions() except +


cdef extern from "primitiv/graph.h":
    cdef CppGraph &CppGraph_get_default "primitiv::Graph::get_default"()
    cdef void CppGraph_set_default "primitiv::Graph::set_default"(CppGraph &g)


cdef class _Node:
    cdef CppNode wrapped


cdef class _Graph:
    cdef CppGraph *wrapped
    cdef CppGraph *wrapped_newed
    cdef object __weakref__


cdef inline _Node wrapNode(CppNode wrapped) except +:
    cdef _Node node = _Node.__new__(_Node)
    node.wrapped = wrapped
    return node


# This is used for holding python instances related to C++.
# Without this variable, python instances are always created when C++ class
# instances are returned from functions.
# It means that users can not compare instances by using "is" operator.
cdef object py_primitiv_graph_weak_dict

cdef inline _Graph wrapGraph(CppGraph *wrapped) except +:
    global py_primitiv_graph_weak_dict

    # _Graph instances should be created and be registered before this
    # function is called.
    return py_primitiv_graph_weak_dict[<uintptr_t> wrapped]
