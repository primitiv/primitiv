#ifndef PRIMITIV_CORE_TYPE_TRAITS_H_
#define PRIMITIV_CORE_TYPE_TRAITS_H_

#include <type_traits>

namespace primitiv {

class Device;
class Graph;
class Node;
class Tensor;

namespace type_traits {

// Returns true_type if Var is Tensor or Node.
template<typename Var> struct is_var : std::false_type {};
template<> struct is_var<Tensor> : std::true_type {};
template<> struct is_var<Node> : std::true_type {};

// Var -> Var
template<typename Var>
using Identity = typename std::enable_if<is_var<Var>::value, Var>::type;

// Container<Var> -> Var
template<typename Container>
using Reduce = typename std::enable_if<
  is_var<typename Container::value_type>::value,
  typename Container::value_type
>::type;

// Container<Var *> -> Var
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

#endif  // PRIMITIV_CORE_TYPE_TRAITS_H_
