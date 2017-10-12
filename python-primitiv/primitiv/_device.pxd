from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from primitiv._tensor cimport CppTensor
from primitiv._shape cimport CppShape


cdef extern from "primitiv/device.h" namespace "primitiv":
    cdef cppclass CppDevice "primitiv::Device":
        enum DeviceType:
            DEVICE_TYPE_CPU = 0x0
            DEVICE_TYPE_CUDA = 0x10000
        CppTensor new_tensor(const CppShape &shape) except +
        CppTensor new_tensor(const CppShape &shape, float k) except +
        # CppTensor new_tensor_by_array(const CppShape &shape, const float values[]) except +
        CppTensor new_tensor_by_vector(const CppShape &shape, const vector[float] &values) except +
        CppTensor copy_tensor(const CppTensor &x) except +
        CppTensor identity(unsigned size) except +
        CppTensor random_bernoulli(const CppShape &shape, float p) except +
        CppTensor random_uniform(const CppShape &shape, float lower, float upper) except +
        CppTensor random_normal(const CppShape &shape, float mean, float sd) except +
        CppTensor random_log_normal(const CppShape &shape, float mean, float sd) except +
        CppTensor pick_fw(const CppTensor &x, const vector[unsigned] &ids, unsigned dim) except +
        CppTensor slice_fw(const CppTensor &x, unsigned dim, unsigned lower, unsigned upper) except +
        CppTensor concat_fw(const vector[const CppTensor *] &xs, unsigned dim) except +
        void pick_bw(const CppTensor &gy, const vector[unsigned] &ids, unsigned dim, CppTensor &gx) except +
        void slice_bw(const CppTensor &gy, unsigned dim, unsigned offset, CppTensor &gx) except +
        CppTensor negate_fw(const CppTensor &x) except +
        CppTensor sqrt_fw(const CppTensor &x) except +
        CppTensor exp_fw(const CppTensor &x) except +
        CppTensor log_fw(const CppTensor &x) except +
        CppTensor tanh_fw(const CppTensor &x) except +
        CppTensor sigmoid_fw(const CppTensor &x) except +
        CppTensor softplus_fw(const CppTensor &x) except +
        CppTensor sin_fw(const CppTensor &x) except +
        CppTensor cos_fw(const CppTensor &x) except +
        CppTensor tan_fw(const CppTensor &x) except +
        CppTensor transpose_fw(const CppTensor &x) except +
        void sqrt_bw(const CppTensor &x, const CppTensor &y, const CppTensor &gy, CppTensor &gx) except +
        void exp_bw(const CppTensor &x, const CppTensor &y, const CppTensor &gy, CppTensor &gx) except +
        void log_bw(const CppTensor &x, const CppTensor &y, const CppTensor &gy, CppTensor &gx) except +
        void tanh_bw(const CppTensor &x, const CppTensor &y, const CppTensor &gy, CppTensor &gx) except +
        void sigmoid_bw(const CppTensor &x, const CppTensor &y, const CppTensor &gy, CppTensor &gx) except +
        void softplus_bw(const CppTensor &x, const CppTensor &y, const CppTensor &gy, CppTensor &gx) except +
        void sin_bw(const CppTensor &x, const CppTensor &y, const CppTensor &gy, CppTensor &gx) except +
        void cos_bw(const CppTensor &x, const CppTensor &y, const CppTensor &gy, CppTensor &gx) except +
        void tan_bw(const CppTensor &x, const CppTensor &y, const CppTensor &gy, CppTensor &gx) except +
        void transpose_bw(const CppTensor &x, const CppTensor &y, const CppTensor &gy, CppTensor &gx) except +
        CppTensor add_const_fw(const CppTensor &x, float k) except +
        CppTensor subtract_const_r_fw(const CppTensor &x, float k) except +
        CppTensor subtract_const_l_fw(const CppTensor &x, float k) except +
        CppTensor multiply_const_fw(const CppTensor &x, float k) except +
        CppTensor divide_const_r_fw(const CppTensor &x, float k) except +
        CppTensor divide_const_l_fw(const CppTensor &x, float k) except +
        CppTensor prelu_fw(const CppTensor &x, float k) except +
        CppTensor elu_fw(const CppTensor &x, float k) except +
        void add_const_bw(const CppTensor &x, const CppTensor &y, const CppTensor &gy, float k, CppTensor &gx) except +
        void subtract_const_r_bw(const CppTensor &x, const CppTensor &y, const CppTensor &gy, float k, CppTensor &gx) except +
        void subtract_const_l_bw(const CppTensor &x, const CppTensor &y, const CppTensor &gy, float k, CppTensor &gx) except +
        void multiply_const_bw(const CppTensor &x, const CppTensor &y, const CppTensor &gy, float k, CppTensor &gx) except +
        void divide_const_r_bw(const CppTensor &x, const CppTensor &y, const CppTensor &gy, float k, CppTensor &gx) except +
        void divide_const_l_bw(const CppTensor &x, const CppTensor &y, const CppTensor &gy, float k, CppTensor &gx) except +
        void prelu_bw(const CppTensor &x, const CppTensor &y, const CppTensor &gy, float k, CppTensor &gx) except +
        void elu_bw(const CppTensor &x, const CppTensor &y, const CppTensor &gy, float k, CppTensor &gx) except +
        CppTensor add_scalar_fw(const CppTensor &x, const CppTensor &k) except +
        CppTensor subtract_scalar_r_fw(const CppTensor &x, const CppTensor &k) except +
        CppTensor subtract_scalar_l_fw(const CppTensor &x, const CppTensor &k) except +
        CppTensor multiply_scalar_fw(const CppTensor &x, const CppTensor &k) except +
        CppTensor divide_scalar_r_fw(const CppTensor &x, const CppTensor &k) except +
        CppTensor divide_scalar_l_fw(const CppTensor &x, const CppTensor &k) except +
        CppTensor add_fw(const CppTensor &a, const CppTensor &b) except +
        CppTensor subtract_fw(const CppTensor &a, const CppTensor &b) except +
        CppTensor multiply_fw(const CppTensor &a, const CppTensor &b) except +
        CppTensor divide_fw(const CppTensor &a, const CppTensor &b) except +
        CppTensor matmul_fw(const CppTensor &a, const CppTensor &b) except +
        void add_bw(const CppTensor &a, const CppTensor &b, const CppTensor &y, const CppTensor &gy, CppTensor &ga, CppTensor &gb) except +
        void subtract_bw(const CppTensor &a, const CppTensor &b, const CppTensor &y, const CppTensor &gy, CppTensor &ga, CppTensor &gb) except +
        void multiply_bw(const CppTensor &a, const CppTensor &b, const CppTensor &y, const CppTensor &gy, CppTensor &ga, CppTensor &gb) except +
        void divide_bw(const CppTensor &a, const CppTensor &b, const CppTensor &y, const CppTensor &gy, CppTensor &ga, CppTensor &gb) except +
        void matmul_bw(const CppTensor &a, const CppTensor &b, const CppTensor &y, const CppTensor &gy, CppTensor &ga, CppTensor &gb) except +
        CppTensor sum_fw(const CppTensor &x, unsigned dim) except +
        CppTensor logsumexp_fw(const CppTensor &x, unsigned dim) except +
        CppTensor broadcast_fw(const CppTensor &x, unsigned dim, unsigned size) except +
        CppTensor batch_sum_fw(const CppTensor &x) except +
        void inplace_multiply_const(float k, CppTensor &x) except +
        void inplace_add(const CppTensor &x, CppTensor &y) except +
        void inplace_subtract(const CppTensor &x, CppTensor &y) except +


cdef extern from "primitiv/device.h" namespace "primitiv::Device":
    cdef CppDevice &get_default()
    cdef void set_default(CppDevice &dev)


cdef class _Device:
    cdef CppDevice *wrapped
    cdef CppDevice *wrapped_newed


cdef inline _Device wrapDevice(CppDevice *wrapped) except +:
    cdef _Device device = _Device.__new__(_Device)
    device.wrapped = wrapped
    return device
