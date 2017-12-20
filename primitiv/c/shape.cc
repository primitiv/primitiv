/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <string>
#include <vector>

#include <primitiv/shape.h>

#include <primitiv/c/internal.h>
#include <primitiv/c/shape.h>

using primitiv::Shape;

extern "C" {

primitiv_Shape *primitiv_Shape_new() {
  return to_c(new Shape);
}
primitiv_Shape *safe_primitiv_Shape_new(primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_new(), status, nullptr);
}

primitiv_Shape *primitiv_Shape_new_with_dims(const uint32_t *dims,
                                             size_t n,
                                             uint32_t batch) {
  return to_c(new Shape(std::vector<uint32_t>(dims, dims + n), batch));
}
primitiv_Shape *safe_primitiv_Shape_new_with_dims(const uint32_t *dims,
                                                  size_t n,
                                                  uint32_t batch,
                                                  primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_new_with_dims(dims, n, batch), status, nullptr);
}

void primitiv_Shape_delete(primitiv_Shape *shape) {
  delete to_cc(shape);
}
void safe_primitiv_Shape_delete(primitiv_Shape *shape,
                                primitiv_Status *status) {
  SAFE_EXPR(primitiv_Shape_delete(shape), status);
}

uint32_t primitiv_Shape_op_getitem(const primitiv_Shape *shape, uint32_t i) {
  return to_cc(shape)->operator[](i);
}
uint32_t safe_primitiv_Shape_op_getitem(const primitiv_Shape *shape,
                                        uint32_t i,
                                        primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_op_getitem(shape, i), status, 0);
}

void primitiv_Shape_dims(const primitiv_Shape *shape, uint32_t *array) {
  std::vector<uint32_t> v = to_cc(shape)->dims();
  std::copy(v.begin(), v.end(), array);
}
void safe_primitiv_Shape_dims(const primitiv_Shape *shape,
                              uint32_t *array,
                              primitiv_Status *status) {
  SAFE_EXPR(primitiv_Shape_dims(shape, array), status);
}

uint32_t primitiv_Shape_depth(const primitiv_Shape *shape) {
  return to_cc(shape)->depth();
}
uint32_t safe_primitiv_Shape_depth(const primitiv_Shape *shape,
                                   primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_depth(shape), status, 0);
}

uint32_t primitiv_Shape_batch(const primitiv_Shape *shape) {
  return to_cc(shape)->batch();
}
uint32_t safe_primitiv_Shape_batch(const primitiv_Shape *shape,
                                   primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_batch(shape), status, 0);
}

uint32_t primitiv_Shape_volume(const primitiv_Shape *shape) {
  return to_cc(shape)->volume();
}
uint32_t safe_primitiv_Shape_volume(const primitiv_Shape *shape,
                                    primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_volume(shape), status, 0);
}

uint32_t primitiv_Shape_lower_volume(const primitiv_Shape *shape,
                                     uint32_t dim) {
  return to_cc(shape)->lower_volume(dim);
}
uint32_t safe_primitiv_Shape_lower_volume(const primitiv_Shape *shape,
                                          uint32_t dim,
                                          primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_lower_volume(shape, dim), status, 0);
}

uint32_t primitiv_Shape_size(const primitiv_Shape *shape) {
  return to_cc(shape)->size();
}
uint32_t safe_primitiv_Shape_size(const primitiv_Shape *shape,
                                  primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_size(shape), status, 0);
}

char *primitiv_Shape_to_string(const primitiv_Shape *shape) {
  std::string str = to_cc(shape)->to_string();
  uint64_t len = str.length();
  auto *c = new char[len + 1];
  std::strncpy(c, str.c_str(), len);
  return c;
}
char *safe_primitiv_Shape_to_string(const primitiv_Shape *shape,
                                    primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_to_string(shape), status, nullptr);
}

bool primitiv_Shape_op_eq(const primitiv_Shape *shape,
                          const primitiv_Shape *rhs) {
  return to_cc(shape)->operator==(*to_cc(rhs));
}
bool safe_primitiv_Shape_op_eq(const primitiv_Shape *shape,
                               const primitiv_Shape *rhs,
                               primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_op_eq(shape, rhs), status, false);
}

bool primitiv_Shape_op_ne(const primitiv_Shape *shape,
                          const primitiv_Shape *rhs) {
  return to_cc(shape)->operator!=(*to_cc(rhs));
}
bool safe_primitiv_Shape_op_ne(const primitiv_Shape *shape,
                               const primitiv_Shape *rhs,
                               primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_op_ne(shape, rhs), status, false);
}

bool primitiv_Shape_has_batch(const primitiv_Shape *shape) {
  return to_cc(shape)->has_batch();
}
bool safe_primitiv_Shape_has_batch(const primitiv_Shape *shape,
                                   primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_has_batch(shape), status, false);
}

bool primitiv_Shape_has_compatible_batch(const primitiv_Shape *shape,
                                         const primitiv_Shape *rhs) {
  return to_cc(shape)->has_compatible_batch(*to_cc(rhs));
}
bool safe_primitiv_Shape_has_compatible_batch(const primitiv_Shape *shape,
                                              const primitiv_Shape *rhs,
                                              primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_has_compatible_batch(shape, rhs), status, false);
}

bool primitiv_Shape_is_scalar(const primitiv_Shape *shape) {
  return to_cc(shape)->is_scalar();
}
bool safe_primitiv_Shape_is_scalar(const primitiv_Shape *shape,
                                   primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_is_scalar(shape), status, false);
}

bool primitiv_Shape_is_row_vector(const primitiv_Shape *shape) {
  return to_cc(shape)->is_row_vector();
}
bool safe_primitiv_Shape_is_row_vector(const primitiv_Shape *shape,
                                       primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_is_row_vector(shape), status, false);
}

bool primitiv_Shape_is_matrix(const primitiv_Shape *shape) {
  return to_cc(shape)->is_matrix();
}
bool safe_primitiv_Shape_is_matrix(const primitiv_Shape *shape,
                                   primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_is_matrix(shape), status, false);
}

bool primitiv_Shape_has_same_dims(const primitiv_Shape *shape,
                                  const primitiv_Shape *rhs) {
  return to_cc(shape)->has_same_dims(*to_cc(rhs));
}
bool safe_primitiv_Shape_has_same_dims(const primitiv_Shape *shape,
                                       const primitiv_Shape *rhs,
                                       primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_has_same_dims(shape, rhs), status, false);
}

bool primitiv_Shape_has_same_loo_dims(const primitiv_Shape *shape,
                                      const primitiv_Shape *rhs,
                                      uint32_t dim) {
  return to_cc(shape)->has_same_loo_dims(*to_cc(rhs), dim);
}
bool safe_primitiv_Shape_has_same_loo_dims(const primitiv_Shape *shape,
                                           const primitiv_Shape *rhs,
                                           uint32_t dim,
                                           primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_has_same_loo_dims(shape, rhs, dim), status, false);
}

primitiv_Shape *primitiv_Shape_resize_dim(const primitiv_Shape *shape,
                                          uint32_t dim,
                                          uint32_t m) {
  return to_c_from_value(to_cc(shape)->resize_dim(dim, m));
}
primitiv_Shape *safe_primitiv_Shape_resize_dim(const primitiv_Shape *shape,
                                               uint32_t dim,
                                               uint32_t m,
                                               primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_resize_dim(shape, dim, m), status, nullptr);
}

primitiv_Shape *primitiv_Shape_resize_batch(const primitiv_Shape *shape,
                                            uint32_t batch) {
  return to_c_from_value(to_cc(shape)->resize_batch(batch));
}
primitiv_Shape *safe_primitiv_Shape_resize_batch(const primitiv_Shape *shape,
                                                 uint32_t batch,
                                                 primitiv_Status *status) {
  SAFE_RETURN(primitiv_Shape_resize_batch(shape, batch), status, nullptr);
}

void primitiv_Shape_update_dim(primitiv_Shape *shape,
                               uint32_t dim,
                               uint32_t m) {
  to_cc(shape)->update_dim(dim, m);
}
void safe_primitiv_Shape_update_dim(primitiv_Shape *shape,
                                    uint32_t dim,
                                    uint32_t m,
                                    primitiv_Status *status) {
  SAFE_EXPR(primitiv_Shape_update_dim(shape, dim, m), status);
}

void primitiv_Shape_update_batch(primitiv_Shape *shape, uint32_t batch) {
  to_cc(shape)->update_batch(batch);
}
void safe_primitiv_Shape_update_batch(primitiv_Shape *shape,
                                      uint32_t batch,
                                      primitiv_Status *status) {
  SAFE_EXPR(primitiv_Shape_update_batch(shape, batch), status);
}

}  // end extern "C"
