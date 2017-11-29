from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool


cdef extern from "primitiv/shape.h":
    cdef cppclass CppShape "primitiv::Shape":
        CppShape() except +
        CppShape(vector[unsigned] &dims, unsigned batch) except +
        unsigned operator[](unsigned i) except +
        vector[unsigned] dims() except +
        unsigned depth() except +
        unsigned batch() except +
        unsigned volume() except +
        unsigned lower_volume(unsigned dim) except +
        unsigned size() except +
        string to_string() except +
        bool operator==(CppShape &rhs) except +
        bool operator!=(CppShape &rhs) except +
        bool has_batch() except +
        bool has_compatible_batch(const CppShape &rhs) except +
        bool is_scalar() except +
        bool is_row_vector() except +
        bool is_matrix() except +
        bool has_same_dims(const CppShape &rhs) except +
        bool has_same_loo_dims(const CppShape &rhs, unsigned dim) except +
        CppShape resize_dim(unsigned dim, unsigned m) except +
        CppShape resize_batch(unsigned batch) except +
        void update_dim(unsigned dim, unsigned m) except +
        void update_batch(unsigned batch) except +


cdef class Shape:
    cdef CppShape wrapped


cdef inline Shape wrapShape(CppShape wrapped) except +:
    cdef Shape shape = Shape.__new__(Shape)
    shape.wrapped = wrapped
    return shape

cdef inline Shape normShape(shapelike):
    if isinstance(shapelike, Shape):
        return shapelike
    else:
        return Shape(shapelike)
