#ifndef PRIMITIV_TYPE_TRAITS_H_
#define PRIMITIV_TYPE_TRAITS_H_

#include <type_traits>

namespace primitiv {

class Node;
class Tensor;

namespace type_traits {

// NOTE(odashi):
// The template variable `Var` could becomes only either `Tensor` or `Node`.

template<typename Var> struct is_var : std::false_type {};
template<> struct is_var<Tensor> : std::true_type {};
template<> struct is_var<Node> : std::true_type {};

template<typename Var>
using Identity = typename std::enable_if<is_var<Var>::value, Var>::type;

template<typename Container>
using Reduce = typename std::enable_if<
  is_var<typename Container::value_type>::value,
  typename Container::value_type
>::type;

template<typename Container>
using ReducePtr = typename std::enable_if<
  std::is_pointer<typename Container::value_type>::value &&
  is_var<
    typename std::remove_const<
      typename std::remove_pointer<typename Container::value_type>::type
    >::type
  >::value,
  typename std::remove_const<
    typename std::remove_pointer<typename Container::value_type>::type
  >::type
>::type;

}  // namespace type_traits

}  // namespace primitiv

#endif  // PRIMITIV_TYPE_TRAITS_H_
