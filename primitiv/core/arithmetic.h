#ifndef PRIMITIV_CORE_ARITHMETIC_H_
#define PRIMITIV_CORE_ARITHMETIC_H_

#include <primitiv/core/basic_functions.h>
#include <primitiv/core/tensor.h>

namespace primitiv {

template<typename Var>
inline type_traits::Identity<Var> operator+(const Var &x) {
  return functions::positive(x);
}

template<typename Var>
inline type_traits::Identity<Var> operator-(const Var &x) {
  return functions::negative(x);
}

template<typename Var>
inline type_traits::Identity<Var> operator+(const Var &x, float k) {
  return functions::add(x, k);
}

template<typename Var>
inline type_traits::Identity<Var> operator+(float k, const Var &x) {
  return functions::add(k, x);
}

template<typename Var>
inline type_traits::Identity<Var> operator+(const Var &a, const Var &b) {
  return functions::add(a, b);
}

template<typename Var>
inline type_traits::Identity<Var> operator-(const Var &x, float k) {
  return functions::subtract(x, k);
}

template<typename Var>
inline type_traits::Identity<Var> operator-(float k, const Var &x) {
  return functions::subtract(k, x);
}

template<typename Var>
inline type_traits::Identity<Var> operator-(const Var &a, const Var &b) {
  return functions::subtract(a, b);
}

template<typename Var>
inline type_traits::Identity<Var> operator*(const Var &x, float k) {
  return functions::multiply(x, k);
}

template<typename Var>
inline type_traits::Identity<Var> operator*(float k, const Var &x) {
  return functions::multiply(k, x);
}

template<typename Var>
inline type_traits::Identity<Var> operator*(const Var &a, const Var &b) {
  return functions::multiply(a, b);
}

template<typename Var>
inline type_traits::Identity<Var> operator/(const Var &x, float k) {
  return functions::divide(x, k);
}

template<typename Var>
inline type_traits::Identity<Var> operator/(float k, const Var &x) {
  return functions::divide(k, x);
}

template<typename Var>
inline type_traits::Identity<Var> operator/(const Var &a, const Var &b) {
  return functions::divide(a, b);
}

inline Tensor &operator*=(Tensor &x, float k) {
  return x.inplace_multiply_const(k);
}

inline Tensor &operator+=(Tensor &x, const Tensor &a) {
  return x.inplace_add(a);
}

inline Tensor &operator-=(Tensor &x, const Tensor &a) {
  return x.inplace_subtract(a);
}

}  // namespace primitiv

#endif  // PRIMITIV_CORE_ARITHMETIC_H_
