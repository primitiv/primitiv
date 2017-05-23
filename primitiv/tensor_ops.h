#ifndef PRIMITIV_TENSOR_OPS_H_
#define PRIMITIV_TENSOR_OPS_H_

#include <primitiv/device.h>
#include <primitiv/tensor.h>

namespace primitiv {

inline Tensor operator+(const Tensor &x) { return x.device()->duplicate(x); }
inline Tensor operator-(const Tensor &x) { return x.device()->negate(x); }
inline Tensor operator+(const Tensor &x, const float k) { return x.device()->add(x, k); }
inline Tensor operator+(const float k, const Tensor &x) { return x.device()->add(x, k); }
inline Tensor operator+(const Tensor &a, const Tensor &b) { return a.device()->add(a, b); }
inline Tensor operator-(const Tensor &x, const float k) { return x.device()->subtract(x, k); }
inline Tensor operator-(const float k, const Tensor &x) { return x.device()->subtract(k, x); }
inline Tensor operator-(const Tensor &a, const Tensor &b) { return a.device()->subtract(a, b); }
inline Tensor operator*(const Tensor &x, const float k) { return x.device()->multiply(x, k); }
inline Tensor operator*(const float k, const Tensor &x) { return x.device()->multiply(x, k); }
inline Tensor operator*(const Tensor &a, const Tensor &b) { return a.device()->multiply(a, b); }
inline Tensor operator/(const Tensor &x, const float k) { return x.device()->divide(x, k); }
inline Tensor operator/(const float k, const Tensor &x) { return x.device()->divide(k, x); }
inline Tensor operator/(const Tensor &a, const Tensor &b) { return a.device()->divide(a, b); }

namespace tensor_ops {

inline Tensor transpose(const Tensor &x) { return x.device()->transpose(x); }
inline Tensor dot(const Tensor &a, const Tensor &b) { return a.device()->dot(a, b); }
inline Tensor exp(const Tensor &x) { return x.device()->exp(x); }
inline Tensor tanh(const Tensor &x) { return x.device()->tanh(x); }

}  // namespace tensor_ops
}  // namespace primitiv

#endif  // PRIMITIV_TENSOR_OPS_H_
