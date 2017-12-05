#include "primitiv_c/internal.h"
#include "primitiv_c/shape.h"

#include <string>
#include <vector>

#include <primitiv/shape.h>

using primitiv::Shape;

extern "C" {

primitiv_Shape *primitiv_Shape_new() {
  return to_c(new Shape);
}

primitiv_Shape *primitiv_Shape_new_with_dims(const uint32_t *dims, size_t n, uint32_t batch) {
  return to_c(new Shape(std::vector<uint32_t>(dims, dims + n), batch));
}

void primitiv_Shape_delete(const primitiv_Shape *shape) {
  delete to_cc(shape);
}

uint32_t primitiv_Shape_op_getitem(const primitiv_Shape *shape, uint32_t i) {
  return to_cc(shape)->operator[](i);
}

const uint32_t *primitiv_Shape_dims(const primitiv_Shape *shape) {
  return &(to_cc(shape)->dims())[0];
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

uint32_t primitiv_Shape_lower_volume(const primitiv_Shape *shape, uint32_t dim) {
  return to_cc(shape)->lower_volume(dim);
}

uint32_t primitiv_Shape_size(const primitiv_Shape *shape) {
  return to_cc(shape)->size();
}

char *primitiv_Shape_to_string(const primitiv_Shape *shape) {
  std::string str = to_cc(shape)->to_string();
  unsigned long len = str.length();
  char *c = new char[len + 1];
  std::strncpy(c, str.c_str(), len);
  return c;
}

bool primitiv_Shape_op_eq(const primitiv_Shape *shape, const primitiv_Shape *rhs) {
  return to_cc(shape)->operator==(*to_cc(rhs));
}

bool primitiv_Shape_op_ne(const primitiv_Shape *shape, const primitiv_Shape *rhs) {
  return to_cc(shape)->operator!=(*to_cc(rhs));
}

bool primitiv_Shape_has_batch(const primitiv_Shape *shape) {
  return to_cc(shape)->has_batch();
}

bool primitiv_Shape_has_compatible_batch(const primitiv_Shape *shape, const primitiv_Shape *rhs) {
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

bool primitiv_Shape_has_same_dims(const primitiv_Shape *shape, const primitiv_Shape *rhs) {
  return to_cc(shape)->has_same_dims(*to_cc(rhs));
}

bool primitiv_Shape_has_same_loo_dims(const primitiv_Shape *shape, const primitiv_Shape *rhs, uint32_t dim) {
  return to_cc(shape)->has_same_loo_dims(*to_cc(rhs), dim);
}

primitiv_Shape *primitiv_Shape_resize_dim(const primitiv_Shape *shape, uint32_t dim, uint32_t m) {
  Shape s = to_cc(shape)->resize_dim(dim, m);
  return to_c_from_value(s);
}

primitiv_Shape *primitiv_Shape_resize_batch(const primitiv_Shape *shape, uint32_t batch) {
  Shape s = to_cc(shape)->resize_batch(batch);
  return to_c_from_value(s);
}

void primitiv_Shape_update_dim(primitiv_Shape *shape, uint32_t dim, uint32_t m) {
  to_cc(shape)->update_dim(dim, m);
}

void primitiv_Shape_update_batch(primitiv_Shape *shape, uint32_t batch) {
  to_cc(shape)->update_batch(batch);
}

}  // end extern "C"
