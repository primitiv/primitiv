from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool


cdef extern from "primitiv/shape.h" namespace "primitiv":
    cdef cppclass Shape:
        Shape() except +
        Shape(vector[unsigned] &dims, unsigned batch) except +
        unsigned operator[](unsigned i) except +
        unsigned depth() except +
        unsigned batch() except +
        unsigned volume() except +
        unsigned lower_volume(unsigned dim) except +
        unsigned size() except +
        string to_string() except +
        bool operator==(Shape &rhs) except +
        bool operator!=(Shape &rhs) except +
        bool has_batch() except +
        bool has_compatible_batch(const Shape &rhs) except +
        bool is_scalar() except +
        bool is_row_vector() except +
        bool is_matrix() except +
        bool has_same_dims(const Shape &rhs) except +
        bool has_same_loo_dims(const Shape &rhs, unsigned dim) except +
        Shape resize_dim(unsigned dim, unsigned m) except +
        Shape resize_batch(unsigned batch) except +
        void update_dim(unsigned dim, unsigned m) except +
        void update_batch(unsigned batch) except +


cdef class _Shape:
    cdef Shape wrapped


cdef inline _Shape wrapShape(Shape wrapped) except +:
    cdef _Shape shape = _Shape.__new__(_Shape)
    shape.wrapped = wrapped
    return shape

cdef inline _Shape normShape(shapelike):
    if isinstance(shapelike, _Shape):
        return shapelike
    else:
        return _Shape(shapelike)
