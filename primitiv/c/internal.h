/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_INTERNAL_H_
#define PRIMITIV_C_INTERNAL_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <primitiv/device.h>
#include <primitiv/error.h>
#include <primitiv/graph.h>
#include <primitiv/initializer.h>
#include <primitiv/model.h>
#include <primitiv/parameter.h>
#include <primitiv/shape.h>
#include <primitiv/tensor.h>
#include <primitiv/optimizer.h>

#include <primitiv/c/status.h>

#define DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(cc_name, c_name) \
inline c_name *to_c(primitiv::cc_name *instance) { \
  return reinterpret_cast<c_name*>(instance); \
} \
inline const c_name *to_c(const primitiv::cc_name *instance) { \
  return reinterpret_cast<const c_name*>(instance); \
} \
inline primitiv::cc_name *to_cc(c_name *instance) { \
  return reinterpret_cast<primitiv::cc_name*>(instance); \
} \
inline const primitiv::cc_name *to_cc(const c_name *instance) { \
  return reinterpret_cast<const primitiv::cc_name*>(instance); \
}

#define DEFINE_VALUE_TO_POINTER_CONVERSION_AS_CAST(cc_name, c_name) \
inline c_name *to_c_from_value(const primitiv::cc_name &instance) { \
  return reinterpret_cast<c_name*>(new primitiv::cc_name(instance)); \
}

namespace primitiv {

void set_status(primitiv_Status *status,
                primitiv_Code code,
                const Error *error);

#define SAFE_EXPR(expr, status) \
try { \
  expr; \
} catch (primitiv::Error &e) { \
  primitiv::set_status(status, primitiv_Code::PRIMITIV_ERROR, &e); \
}

#define SAFE_RETURN(expr, status, default) \
try { \
  return expr; \
} catch (primitiv::Error &e) { \
  primitiv::set_status(status, primitiv_Code::PRIMITIV_ERROR, &e); \
} \
return default

}  // namespace primitiv

struct primitiv_Device;

DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(Device, primitiv_Device);

struct primitiv_Node;

DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(Node, primitiv_Node);
DEFINE_VALUE_TO_POINTER_CONVERSION_AS_CAST(Node, primitiv_Node);

struct primitiv_Graph;

DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(Graph, primitiv_Graph);

struct primitiv_Initializer;

DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(Initializer, primitiv_Initializer);

struct primitiv_Model;

DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(Model, primitiv_Model);

struct primitiv_Parameter;

DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(Parameter, primitiv_Parameter);

struct primitiv_Shape;

DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(Shape, primitiv_Shape);
DEFINE_VALUE_TO_POINTER_CONVERSION_AS_CAST(Shape, primitiv_Shape);

struct primitiv_Tensor;

DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(Tensor, primitiv_Tensor);
DEFINE_VALUE_TO_POINTER_CONVERSION_AS_CAST(Tensor, primitiv_Tensor);

struct primitiv_Optimizer;

DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(Optimizer, primitiv_Optimizer);

template<typename T>
class StrValMap : public std::unordered_map<std::string, T> {
 public:
  StrValMap() : keys_{}, values_() {}

  std::vector<const char*> &keys() {
    keys_.clear();
    for (auto it = this->begin(); it != this->end(); ++it) {
      keys_.push_back(it->first.c_str());
    }
    return keys_;
  }

  std::vector<T> &values() {
    values_.clear();
    for (auto it = this->begin(); it != this->end(); ++it) {
      values_.push_back(it->second);
    }
    return values_;
  }

 protected:
  std::vector<const char*> keys_;
  std::vector<T> values_;
};

struct primitiv_StrIntMap;
typedef StrValMap<std::uint32_t> StrIntMap;

inline primitiv_StrIntMap *to_c(StrIntMap *instance) {
  return reinterpret_cast<primitiv_StrIntMap*>(instance);
}
inline const primitiv_StrIntMap *to_c(const StrIntMap *instance) {
  return reinterpret_cast<const primitiv_StrIntMap*>(instance);
}
inline StrIntMap *to_cc(primitiv_StrIntMap *instance) {
  return reinterpret_cast<StrIntMap*>(instance);
}
inline const StrIntMap *to_cc(const primitiv_StrIntMap *instance) {
  return reinterpret_cast<const StrIntMap*>(instance);
}

struct primitiv_StrFloatMap;
typedef StrValMap<float> StrFloatMap;

inline primitiv_StrFloatMap *to_c(StrFloatMap *instance) {
  return reinterpret_cast<primitiv_StrFloatMap*>(instance);
}
inline const primitiv_StrFloatMap *to_c(const StrFloatMap *instance) {
  return reinterpret_cast<const primitiv_StrFloatMap*>(instance);
}
inline StrFloatMap *to_cc(primitiv_StrFloatMap *instance) {
  return reinterpret_cast<StrFloatMap*>(instance);
}
inline const StrFloatMap *to_cc(const primitiv_StrFloatMap *instance) {
  return reinterpret_cast<const StrFloatMap*>(instance);
}

#endif  // PRIMITIV_C_INTERNAL_H_
