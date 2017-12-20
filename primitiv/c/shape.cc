/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <config.h>

#include <string>
#include <vector>

#include <primitiv/shape.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/shape.h>

using primitiv::Shape;
using primitiv::c::internal::to_c;
using primitiv::c::internal::to_cc;
using primitiv::c::internal::to_c_from_value;

extern "C" {

primitiv_Shape *primitiv_Shape_new() {
  return to_c(new Shape());
}

primitiv_Status primitiv_Shape_new_with_dims(primitiv_Shape **shape,
                                             const uint32_t *dims,
                                             size_t n,
                                             uint32_t batch) {
  try {
    *shape = to_c(new Shape(std::vector<uint32_t>(dims, dims + n), batch));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

void primitiv_Shape_delete(primitiv_Shape *shape) {
  delete to_cc(shape);
}

uint32_t primitiv_Shape_op_getitem(const primitiv_Shape *shape, uint32_t i) {
  return to_cc(shape)->operator[](i);
}

void primitiv_Shape_dims(const primitiv_Shape *shape, uint32_t *array) {
  std::vector<uint32_t> v = to_cc(shape)->dims();
  std::copy(v.begin(), v.end(), array);
}

uint32_t primitiv_Shape_depth(const primitiv_Shape *shape) {
  return to_cc(shape)->depth();
}

uint32_t primitiv_Shape_batch(const primitiv_Shape *shape) {
  return to_cc(shape)->batch();
}

uint32_t primitiv_Shape_volume(const primitiv_Shape *shape) {
  return to_cc(shape)->volume();
}

uint32_t primitiv_Shape_lower_volume(const primitiv_Shape *shape,
                                     uint32_t dim) {
  return to_cc(shape)->lower_volume(dim);
}

uint32_t primitiv_Shape_size(const primitiv_Shape *shape) {
  return to_cc(shape)->size();
}

char *primitiv_Shape_to_string(const primitiv_Shape *shape) {
  std::string str = to_cc(shape)->to_string();
  uint64_t len = str.length();
  auto *c = new char[len + 1];
  std::strncpy(c, str.c_str(), len);
  return c;
}

bool primitiv_Shape_op_eq(const primitiv_Shape *shape,
                          const primitiv_Shape *rhs) {
  return to_cc(shape)->operator==(*to_cc(rhs));
}

bool primitiv_Shape_op_ne(const primitiv_Shape *shape,
                          const primitiv_Shape *rhs) {
  return to_cc(shape)->operator!=(*to_cc(rhs));
}

bool primitiv_Shape_has_batch(const primitiv_Shape *shape) {
  return to_cc(shape)->has_batch();
}

bool primitiv_Shape_has_compatible_batch(const primitiv_Shape *shape,
                                         const primitiv_Shape *rhs) {
  return to_cc(shape)->has_compatible_batch(*to_cc(rhs));
}

bool primitiv_Shape_is_scalar(const primitiv_Shape *shape) {
  return to_cc(shape)->is_scalar();
}

bool primitiv_Shape_is_row_vector(const primitiv_Shape *shape) {
  return to_cc(shape)->is_row_vector();
}

bool primitiv_Shape_is_matrix(const primitiv_Shape *shape) {
  return to_cc(shape)->is_matrix();
}

bool primitiv_Shape_has_same_dims(const primitiv_Shape *shape,
                                  const primitiv_Shape *rhs) {
  return to_cc(shape)->has_same_dims(*to_cc(rhs));
}

bool primitiv_Shape_has_same_loo_dims(const primitiv_Shape *shape,
                                      const primitiv_Shape *rhs,
                                      uint32_t dim) {
  return to_cc(shape)->has_same_loo_dims(*to_cc(rhs), dim);
}

primitiv_Status primitiv_Shape_resize_dim(const primitiv_Shape *shape,
                                          uint32_t dim,
                                          uint32_t m,
                                          primitiv_Shape **new_shape) {
  try {
    *new_shape = to_c_from_value(to_cc(shape)->resize_dim(dim, m));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Shape_resize_batch(const primitiv_Shape *shape,
                                            uint32_t batch,
                                            primitiv_Shape **new_shape) {
  try {
    *new_shape = to_c_from_value(to_cc(shape)->resize_batch(batch));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Shape_update_dim(primitiv_Shape *shape,
                                          uint32_t dim, uint32_t m) {
  try {
    to_cc(shape)->update_dim(dim, m);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Shape_update_batch(primitiv_Shape *shape,
                                            uint32_t batch) {
  try {
    to_cc(shape)->update_batch(batch);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

}  // end extern "C"
