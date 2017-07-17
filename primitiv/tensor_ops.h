#ifndef PRIMITIV_TENSOR_OPS_H_
#define PRIMITIV_TENSOR_OPS_H_

#include <vector>
#include <primitiv/device.h>
#include <primitiv/error.h>
#include <primitiv/tensor.h>

namespace primitiv {

Tensor operator+(const Tensor &x);
Tensor operator-(const Tensor &x);
Tensor operator+(const Tensor &x, float k);
Tensor operator+(float k, const Tensor &x);
Tensor operator+(const Tensor &a, const Tensor &b);
Tensor operator-(const Tensor &x, float k);
Tensor operator-(float k, const Tensor &x);
Tensor operator-(const Tensor &a, const Tensor &b);
Tensor operator*(const Tensor &x, float k);
Tensor operator*(float k, const Tensor &x);
Tensor operator*(const Tensor &a, const Tensor &b);
Tensor operator/(const Tensor &x, float k);
Tensor operator/(float k, const Tensor &x);
Tensor operator/(const Tensor &a, const Tensor &b);

namespace tensor_ops {

Tensor copy(const Tensor &x, Device &dev);

Tensor pick(const Tensor &x, unsigned dim, const std::vector<unsigned> &ids);
Tensor slice(const Tensor &x, unsigned dim, unsigned lower, unsigned upper);
Tensor concat(const std::vector<const Tensor *> &xs, unsigned dim);

Tensor reshape(const Tensor &x, const Shape &new_shape);
Tensor flatten(const Tensor &x);

Tensor transpose(const Tensor &x);
Tensor matmul(const Tensor &a, const Tensor &b);
Tensor sqrt(const Tensor &x);
Tensor exp(const Tensor &x);
Tensor tanh(const Tensor &x);
Tensor sigmoid(const Tensor &x);
Tensor softplus(const Tensor &x);
Tensor sin(const Tensor &x);
Tensor cos(const Tensor &x);
Tensor tan(const Tensor &x);
Tensor relu(const Tensor &x);
Tensor lrelu(const Tensor &x);
Tensor prelu(const Tensor &x, float a);
Tensor elu(const Tensor &x, float a);
Tensor selu(
    const Tensor &x,
    float a = 1.6732632423543772848170429916717,
    float s = 1.0507009873554804934193349852946);

Tensor sum(const Tensor &x, unsigned dim);
Tensor logsumexp(const Tensor &x, unsigned dim);
Tensor log_softmax(const Tensor &x, unsigned dim);
Tensor softmax(const Tensor &x, unsigned dim);
Tensor broadcast(const Tensor &x, unsigned dim, unsigned size);

Tensor batch_sum(const Tensor &x);

Tensor softmax_cross_entropy(const Tensor &x, const Tensor &t, unsigned dim);
Tensor softmax_cross_entropy(const Tensor &x, unsigned dim, const std::vector<unsigned> &ids);

}  // namespace tensor_ops
}  // namespace primitiv

#endif  // PRIMITIV_TENSOR_OPS_H_
