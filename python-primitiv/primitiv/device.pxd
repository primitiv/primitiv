from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from primitiv.tensor cimport Tensor
from primitiv.shape cimport Shape


cdef extern from "primitiv/device.h" namespace "primitiv":
    cdef cppclass Device:
        enum DeviceType:
            DEVICE_TYPE_CPU = 0x0
            DEVICE_TYPE_CUDA = 0x10000
        Tensor new_tensor(const Shape &shape) except +
        Tensor new_tensor(const Shape &shape, float k) except +
        # Tensor new_tensor_by_array(const Shape &shape, const float values[]) except +
        Tensor new_tensor_by_vector(const Shape &shape, const vector[float] &values) except +
        Tensor copy_tensor(const Tensor &x) except +
        Tensor identity(unsigned size) except +
        Tensor random_bernoulli(const Shape &shape, float p) except +
        Tensor random_uniform(const Shape &shape, float lower, float upper) except +
        Tensor random_normal(const Shape &shape, float mean, float sd) except +
        Tensor random_log_normal(const Shape &shape, float mean, float sd) except +
        Tensor pick_fw(const Tensor &x, const vector[unsigned] &ids, unsigned dim) except +
        Tensor slice_fw(const Tensor &x, unsigned dim, unsigned lower, unsigned upper) except +
        Tensor concat_fw(const vector[const Tensor *] &xs, unsigned dim) except +
        void pick_bw(const Tensor &gy, const vector[unsigned] &ids, unsigned dim, Tensor &gx) except +
        void slice_bw(const Tensor &gy, unsigned dim, unsigned offset, Tensor &gx) except +
        Tensor negate_fw(const Tensor &x) except +
        Tensor sqrt_fw(const Tensor &x) except +
        Tensor exp_fw(const Tensor &x) except +
        Tensor log_fw(const Tensor &x) except +
        Tensor tanh_fw(const Tensor &x) except +
        Tensor sigmoid_fw(const Tensor &x) except +
        Tensor softplus_fw(const Tensor &x) except +
        Tensor sin_fw(const Tensor &x) except +
        Tensor cos_fw(const Tensor &x) except +
        Tensor tan_fw(const Tensor &x) except +
        Tensor transpose_fw(const Tensor &x) except +
        void sqrt_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) except +
        void exp_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) except +
        void log_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) except +
        void tanh_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) except +
        void sigmoid_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) except +
        void softplus_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) except +
        void sin_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) except +
        void cos_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) except +
        void tan_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) except +
        void transpose_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) except +
        Tensor add_const_fw(const Tensor &x, float k) except +
        Tensor subtract_const_r_fw(const Tensor &x, float k) except +
        Tensor subtract_const_l_fw(const Tensor &x, float k) except +
        Tensor multiply_const_fw(const Tensor &x, float k) except +
        Tensor divide_const_r_fw(const Tensor &x, float k) except +
        Tensor divide_const_l_fw(const Tensor &x, float k) except +
        Tensor prelu_fw(const Tensor &x, float k) except +
        Tensor elu_fw(const Tensor &x, float k) except +
        void add_const_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) except +
        void subtract_const_r_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) except +
        void subtract_const_l_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) except +
        void multiply_const_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) except +
        void divide_const_r_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) except +
        void divide_const_l_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) except +
        void prelu_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) except +
        void elu_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) except +
        Tensor add_scalar_fw(const Tensor &x, const Tensor &k) except +
        Tensor subtract_scalar_r_fw(const Tensor &x, const Tensor &k) except +
        Tensor subtract_scalar_l_fw(const Tensor &x, const Tensor &k) except +
        Tensor multiply_scalar_fw(const Tensor &x, const Tensor &k) except +
        Tensor divide_scalar_r_fw(const Tensor &x, const Tensor &k) except +
        Tensor divide_scalar_l_fw(const Tensor &x, const Tensor &k) except +
        Tensor add_fw(const Tensor &a, const Tensor &b) except +
        Tensor subtract_fw(const Tensor &a, const Tensor &b) except +
        Tensor multiply_fw(const Tensor &a, const Tensor &b) except +
        Tensor divide_fw(const Tensor &a, const Tensor &b) except +
        Tensor matmul_fw(const Tensor &a, const Tensor &b) except +
        void add_bw(const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy, Tensor &ga, Tensor &gb) except +
        void subtract_bw(const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy, Tensor &ga, Tensor &gb) except +
        void multiply_bw(const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy, Tensor &ga, Tensor &gb) except +
        void divide_bw(const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy, Tensor &ga, Tensor &gb) except +
        void matmul_bw(const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy, Tensor &ga, Tensor &gb) except +
        Tensor sum_fw(const Tensor &x, unsigned dim) except +
        Tensor logsumexp_fw(const Tensor &x, unsigned dim) except +
        Tensor broadcast_fw(const Tensor &x, unsigned dim, unsigned size) except +
        Tensor batch_sum_fw(const Tensor &x) except +
        void inplace_multiply_const(float k, Tensor &x) except +
        void inplace_add(const Tensor &x, Tensor &y) except +
        void inplace_subtract(const Tensor &x, Tensor &y) except +


cdef class _Device:
    cdef Device *wrapped
    cdef Device *wrapped_newed


cdef inline _Device wrapDevice(Device *wrapped) except +:
    cdef _Device device = _Device.__new__(_Device)
    device.wrapped = wrapped
    return device
