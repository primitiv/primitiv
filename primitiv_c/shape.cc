#include "primitiv_c/internal.h"
#include "primitiv_c/shape.h"

#include <string>
#include <vector>

#include <primitiv/shape.h>

using primitiv::Shape;

extern "C" {

primitiv_Shape *primitiv_Shape_new() {
  return new primitiv_Shape;
}

primitiv_Shape *primitiv_Shape_new_with_dims(const uint32_t *dims, size_t n) {
  return new primitiv_Shape{std::vector<uint32_t>(dims, dims + n)};
}

primitiv_Shape *primitiv_Shape_new_with_dims_batch(const uint32_t *dims, size_t n, uint32_t batch) {
  return new primitiv_Shape{Shape{std::vector<uint32_t>(dims, dims + n), batch}};
}

void primitiv_Shape_delete(const primitiv_Shape *shape) {
  delete shape;
}

uint32_t primitiv_Shape_op_getitem(const primitiv_Shape *shape, uint32_t i) {
  return shape->shape.operator[](i);
}

const uint32_t *primitiv_Shape_dims(const primitiv_Shape *shape) {
  return &(shape->shape.dims())[0];
}

uint32_t primitiv_Shape_depth(const primitiv_Shape *shape) {
  return shape->shape.depth();
}

uint32_t primitiv_Shape_batch(const primitiv_Shape *shape) {
  return shape->shape.batch();
}

uint32_t primitiv_Shape_volume(const primitiv_Shape *shape) {
  return shape->shape.volume();
}

uint32_t primitiv_Shape_lower_volume(const primitiv_Shape *shape, uint32_t dim) {
  return shape->shape.lower_volume(dim);
}

uint32_t primitiv_Shape_size(const primitiv_Shape *shape) {
  return shape->shape.size();
}

char *primitiv_Shape_to_string(const primitiv_Shape *shape) {
  std::string str = shape->shape.to_string();
  unsigned long len = str.length();
  char *c = new char[len + 1];
  std::strncpy(c, str.c_str(), len);
  return c;
}

bool primitiv_Shape_op_eq(const primitiv_Shape *shape, const primitiv_Shape *rhs) {
  return shape->shape.operator==(rhs->shape);
}

bool primitiv_Shape_op_ne(const primitiv_Shape *shape, const primitiv_Shape *rhs) {
  return shape->shape.operator!=(rhs->shape);
}

bool primitiv_Shape_has_batch(const primitiv_Shape *shape) {
  return shape->shape.has_batch();
}

bool primitiv_Shape_has_compatible_batch(const primitiv_Shape *shape, const primitiv_Shape *rhs) {
  return shape->shape.has_compatible_batch(rhs->shape);
}

bool primitiv_Shape_is_scalar(const primitiv_Shape *shape) {
  return shape->shape.is_scalar();
}

bool primitiv_Shape_is_row_vector(const primitiv_Shape *shape) {
  return shape->shape.is_row_vector();
}

bool primitiv_Shape_is_matrix(const primitiv_Shape *shape) {
  return shape->shape.is_matrix();
}

bool primitiv_Shape_has_same_dims(const primitiv_Shape *shape, const primitiv_Shape *rhs) {
  return shape->shape.has_same_dims(rhs->shape);
}

bool primitiv_Shape_has_same_loo_dims(const primitiv_Shape *shape, const primitiv_Shape *rhs, uint32_t dim) {
  return shape->shape.has_same_loo_dims(rhs->shape, dim);
}

primitiv_Shape *primitiv_Shape_resize_dim(const primitiv_Shape *shape, uint32_t dim, uint32_t m) {
  return new primitiv_Shape{shape->shape.resize_dim(dim, m)};
}

primitiv_Shape *primitiv_Shape_resize_batch(const primitiv_Shape *shape, uint32_t batch) {
  return new primitiv_Shape{shape->shape.resize_batch(batch)};
}

void primitiv_Shape_update_dim(primitiv_Shape *shape, uint32_t dim, uint32_t m) {
  shape->shape.update_dim(dim, m);
}

void primitiv_Shape_update_batch(primitiv_Shape *shape, uint32_t batch) {
  shape->shape.update_batch(batch);
}

}  // end extern "C"
