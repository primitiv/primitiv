from libcpp.vector cimport vector

from primitiv.shape cimport normShape
from primitiv.tensor cimport wrapTensor, _Tensor


cdef class _Device:

    def new_tensor(self, shape, float k = 0):
        return wrapTensor(self.wrapped.new_tensor(normShape(shape).wrapped, k))

    #def new_tensor_by_array(self, shape, float values[])
        #return wrapTensor(self.wrapped.new_tensor_by_array(const Shape &shape, const float values[]))

    def new_tensor_by_vector(self, shape, vector[float] values):
        return wrapTensor(self.wrapped.new_tensor_by_vector(normShape(shape).wrapped, values))

    def copy_tensor(self, _Tensor x):
        return wrapTensor(self.wrapped.copy_tensor(x.wrapped))

    def identity(self, unsigned size):
        return wrapTensor(self.wrapped.identity(size))

    def random_bernoulli(self, shape, float p):
        return wrapTensor(self.wrapped.random_bernoulli(normShape(shape).wrapped, p))

    def random_uniform(self, shape, float lower, float upper):
        return wrapTensor(self.wrapped.random_uniform(normShape(shape).wrapped, lower, upper))

    def random_normal(self, shape, float mean, float sd):
        return wrapTensor(self.wrapped.random_normal(normShape(shape).wrapped, mean, sd))

    def random_log_normal(self, shape, float mean, float sd):
        return wrapTensor(self.wrapped.random_log_normal(normShape(shape).wrapped, mean, sd))

    def pick_fw(self, _Tensor x, vector[unsigned] ids, unsigned dim):
        return wrapTensor(self.wrapped.pick_fw(x.wrapped, ids, dim))

    def slice_fw(self, _Tensor x, unsigned dim, unsigned lower, unsigned upper):
        return wrapTensor(self.wrapped.slice_fw(x.wrapped, dim, lower, upper))

    #def concat_fw(self, vector[const Tensor *] xs, unsigned dim):
    #    return wrapTensor(self.wrapped.concat_fw(xs, dim))

    def pick_bw(self, _Tensor gy, vector[unsigned] ids, unsigned dim, _Tensor gx):
        self.wrapped.pick_bw(gy.wrapped, ids, dim, gx.wrapped)
        return

    def slice_bw(self, _Tensor gy, unsigned dim, unsigned offset, _Tensor gx):
        self.wrapped.slice_bw(gy.wrapped, dim, offset, gx.wrapped)
        return

    def negate_fw(self, _Tensor x):
        return wrapTensor(self.wrapped.negate_fw(x.wrapped))

    def sqrt_fw(self, _Tensor x):
        return wrapTensor(self.wrapped.sqrt_fw(x.wrapped))

    def exp_fw(self, _Tensor x):
        return wrapTensor(self.wrapped.exp_fw(x.wrapped))

    def log_fw(self, _Tensor x):
        return wrapTensor(self.wrapped.log_fw(x.wrapped))

    def tanh_fw(self, _Tensor x):
        return wrapTensor(self.wrapped.tanh_fw(x.wrapped))

    def sigmoid_fw(self, _Tensor x):
        return wrapTensor(self.wrapped.sigmoid_fw(x.wrapped))

    def softplus_fw(self, _Tensor x):
        return wrapTensor(self.wrapped.softplus_fw(x.wrapped))

    def sin_fw(self, _Tensor x):
        return wrapTensor(self.wrapped.sin_fw(x.wrapped))

    def cos_fw(self, _Tensor x):
        return wrapTensor(self.wrapped.cos_fw(x.wrapped))

    def tan_fw(self, _Tensor x):
        return wrapTensor(self.wrapped.tan_fw(x.wrapped))

    def transpose_fw(self, _Tensor x):
        return wrapTensor(self.wrapped.transpose_fw(x.wrapped))

    def sqrt_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.wrapped.sqrt_bw(x.wrapped, y.wrapped, gy.wrapped, gx.wrapped)
        return

    def exp_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.wrapped.exp_bw(x.wrapped, y.wrapped, gy.wrapped, gx.wrapped)
        return

    def log_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.wrapped.log_bw(x.wrapped, y.wrapped, gy.wrapped, gx.wrapped)
        return

    def tanh_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.wrapped.tanh_bw(x.wrapped, y.wrapped, gy.wrapped, gx.wrapped)
        return

    def sigmoid_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.wrapped.sigmoid_bw(x.wrapped, y.wrapped, gy.wrapped, gx.wrapped)
        return

    def softplus_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.wrapped.softplus_bw(x.wrapped, y.wrapped, gy.wrapped, gx.wrapped)
        return

    def sin_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.wrapped.sin_bw(x.wrapped, y.wrapped, gy.wrapped, gx.wrapped)
        return

    def cos_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.wrapped.cos_bw(x.wrapped, y.wrapped, gy.wrapped, gx.wrapped)
        return

    def tan_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.wrapped.tan_bw(x.wrapped, y.wrapped, gy.wrapped, gx.wrapped)
        return

    def transpose_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.wrapped.transpose_bw(x.wrapped, y.wrapped, gy.wrapped, gx.wrapped)
        return

    def add_const_fw(self, _Tensor x, float k):
        return wrapTensor(self.wrapped.add_const_fw(x.wrapped, k))

    def subtract_const_r_fw(self, _Tensor x, float k):
        return wrapTensor(self.wrapped.subtract_const_r_fw(x.wrapped, k))

    def subtract_const_l_fw(self, _Tensor x, float k):
        return wrapTensor(self.wrapped.subtract_const_l_fw(x.wrapped, k))

    def multiply_const_fw(self, _Tensor x, float k):
        return wrapTensor(self.wrapped.multiply_const_fw(x.wrapped, k))

    def divide_const_r_fw(self, _Tensor x, float k):
        return wrapTensor(self.wrapped.divide_const_r_fw(x.wrapped, k))

    def divide_const_l_fw(self, _Tensor x, float k):
        return wrapTensor(self.wrapped.divide_const_l_fw(x.wrapped, k))

    def prelu_fw(self, _Tensor x, float k):
        return wrapTensor(self.wrapped.prelu_fw(x.wrapped, k))

    def elu_fw(self, _Tensor x, float k):
        return wrapTensor(self.wrapped.elu_fw(x.wrapped, k))

    def add_const_bw(self, _Tensor x, _Tensor y, _Tensor gy, float k, _Tensor gx):
        self.wrapped.add_const_bw(x.wrapped, y.wrapped, gy.wrapped, k, gx.wrapped)
        return

    def subtract_const_r_bw(self, _Tensor x, _Tensor y, _Tensor gy, float k, _Tensor gx):
        self.wrapped.subtract_const_r_bw(x.wrapped, y.wrapped, gy.wrapped, k, gx.wrapped)
        return

    def subtract_const_l_bw(self, _Tensor x, _Tensor y, _Tensor gy, float k, _Tensor gx):
        self.wrapped.subtract_const_l_bw(x.wrapped, y.wrapped, gy.wrapped, k, gx.wrapped)
        return

    def multiply_const_bw(self, _Tensor x, _Tensor y, _Tensor gy, float k, _Tensor gx):
        self.wrapped.multiply_const_bw(x.wrapped, y.wrapped, gy.wrapped, k, gx.wrapped)
        return

    def divide_const_r_bw(self, _Tensor x, _Tensor y, _Tensor gy, float k, _Tensor gx):
        self.wrapped.divide_const_r_bw(x.wrapped, y.wrapped, gy.wrapped, k, gx.wrapped)
        return

    def divide_const_l_bw(self, _Tensor x, _Tensor y, _Tensor gy, float k, _Tensor gx):
        self.wrapped.divide_const_l_bw(x.wrapped, y.wrapped, gy.wrapped, k, gx.wrapped)
        return

    def prelu_bw(self, _Tensor x, _Tensor y, _Tensor gy, float k, _Tensor gx):
        self.wrapped.prelu_bw(x.wrapped, y.wrapped, gy.wrapped, k, gx.wrapped)
        return

    def elu_bw(self, _Tensor x, _Tensor y, _Tensor gy, float k, _Tensor gx):
        self.wrapped.elu_bw(x.wrapped, y.wrapped, gy.wrapped, k, gx.wrapped)
        return

    def add_scalar_fw(self, _Tensor x, _Tensor k):
        return wrapTensor(self.wrapped.add_scalar_fw(x.wrapped, k.wrapped))

    def subtract_scalar_r_fw(self, _Tensor x, _Tensor k):
        return wrapTensor(self.wrapped.subtract_scalar_r_fw(x.wrapped, k.wrapped))

    def subtract_scalar_l_fw(self, _Tensor x, _Tensor k):
        return wrapTensor(self.wrapped.subtract_scalar_l_fw(x.wrapped, k.wrapped))

    def multiply_scalar_fw(self, _Tensor x, _Tensor k):
        return wrapTensor(self.wrapped.multiply_scalar_fw(x.wrapped, k.wrapped))

    def divide_scalar_r_fw(self, _Tensor x, _Tensor k):
        return wrapTensor(self.wrapped.divide_scalar_r_fw(x.wrapped, k.wrapped))

    def divide_scalar_l_fw(self, _Tensor x, _Tensor k):
        return wrapTensor(self.wrapped.divide_scalar_l_fw(x.wrapped, k.wrapped))

    def add_fw(self, _Tensor a, _Tensor b):
        return wrapTensor(self.wrapped.add_fw(a.wrapped, b.wrapped))

    def subtract_fw(self, _Tensor a, _Tensor b):
        return wrapTensor(self.wrapped.subtract_fw(a.wrapped, b.wrapped))

    def multiply_fw(self, _Tensor a, _Tensor b):
        return wrapTensor(self.wrapped.multiply_fw(a.wrapped, b.wrapped))

    def divide_fw(self, _Tensor a, _Tensor b):
        return wrapTensor(self.wrapped.divide_fw(a.wrapped, b.wrapped))

    def matmul_fw(self, _Tensor a, _Tensor b):
        return wrapTensor(self.wrapped.matmul_fw(a.wrapped, b.wrapped))

    def add_bw(self, _Tensor a, _Tensor b, _Tensor y, _Tensor gy, _Tensor ga, _Tensor gb):
        self.wrapped.add_bw(a.wrapped, b.wrapped, y.wrapped, gy.wrapped, ga.wrapped, gb.wrapped)
        return

    def subtract_bw(self, _Tensor a, _Tensor b, _Tensor y, _Tensor gy, _Tensor ga, _Tensor gb):
        self.wrapped.subtract_bw(a.wrapped, b.wrapped, y.wrapped, gy.wrapped, ga.wrapped, gb.wrapped)
        return

    def multiply_bw(self, _Tensor a, _Tensor b, _Tensor y, _Tensor gy, _Tensor ga, _Tensor gb):
        self.wrapped.multiply_bw(a.wrapped, b.wrapped, y.wrapped, gy.wrapped, ga.wrapped, gb.wrapped)
        return

    def divide_bw(self, _Tensor a, _Tensor b, _Tensor y, _Tensor gy, _Tensor ga, _Tensor gb):
        self.wrapped.divide_bw(a.wrapped, b.wrapped, y.wrapped, gy.wrapped, ga.wrapped, gb.wrapped)
        return

    def matmul_bw(self, _Tensor a, _Tensor b, _Tensor y, _Tensor gy, _Tensor ga, _Tensor gb):
        self.wrapped.matmul_bw(a.wrapped, b.wrapped, y.wrapped, gy.wrapped, ga.wrapped, gb.wrapped)
        return

    def sum_fw(self, _Tensor x, unsigned dim):
        return wrapTensor(self.wrapped.sum_fw(x.wrapped, dim))

    def logsumexp_fw(self, _Tensor x, unsigned dim):
        return wrapTensor(self.wrapped.logsumexp_fw(x.wrapped, dim))

    def broadcast_fw(self, _Tensor x, unsigned dim, unsigned size):
        return wrapTensor(self.wrapped.broadcast_fw(x.wrapped, dim, size))

    def batch_sum_fw(self, _Tensor x):
        return wrapTensor(self.wrapped.batch_sum_fw(x.wrapped))

    def inplace_multiply_const(self, float k, _Tensor x):
        self.wrapped.inplace_multiply_const(k, x.wrapped)
        return

    def inplace_add(self, _Tensor x, _Tensor y):
        self.wrapped.inplace_add(x.wrapped, y.wrapped)
        return

    def inplace_subtract(self, _Tensor x, _Tensor y):
        self.wrapped.inplace_subtract(x.wrapped, y.wrapped)
        return
