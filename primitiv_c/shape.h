#ifndef PRIMITIV_C_SHAPE_H_
#define PRIMITIV_C_SHAPE_H_

#include "primitiv_c/define.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct primitiv_Shape primitiv_Shape;

primitiv_Shape *primitiv_Shape_new();

primitiv_Shape *primitiv_Shape_new_with_dims(const uint32_t *dims, size_t n);

primitiv_Shape *primitiv_Shape_new_with_dims_batch(const uint32_t *dims, size_t n, uint32_t batch);

void primitiv_Shape_delete(const primitiv_Shape *shape);

uint32_t primitiv_Shape_op_getitem(const primitiv_Shape *shape, uint32_t i);

const uint32_t *primitiv_Shape_dims(const primitiv_Shape *shape);

uint32_t primitiv_Shape_depth(const primitiv_Shape *shape);

uint32_t primitiv_Shape_batch(const primitiv_Shape *shape);

uint32_t primitiv_Shape_volume(const primitiv_Shape *shape);

uint32_t primitiv_Shape_lower_volume(const primitiv_Shape *shape, uint32_t dim);

uint32_t primitiv_Shape_size(const primitiv_Shape *shape);

char *primitiv_Shape_to_string(const primitiv_Shape *shape);

bool primitiv_Shape_op_eq(const primitiv_Shape *shape, const primitiv_Shape *rhs);

bool primitiv_Shape_op_ne(const primitiv_Shape *shape, const primitiv_Shape *rhs);

bool primitiv_Shape_has_batch(const primitiv_Shape *shape);

bool primitiv_Shape_has_compatible_batch(const primitiv_Shape *shape, const primitiv_Shape *rhs);

bool primitiv_Shape_is_scalar(const primitiv_Shape *shape);

bool primitiv_Shape_is_row_vector(const primitiv_Shape *shape);

bool primitiv_Shape_is_matrix(const primitiv_Shape *shape);

bool primitiv_Shape_has_same_dims(const primitiv_Shape *shape, const primitiv_Shape *rhs);

bool primitiv_Shape_has_same_loo_dims(const primitiv_Shape *shape, const primitiv_Shape *rhs, uint32_t dim);

primitiv_Shape *primitiv_Shape_resize_dim(const primitiv_Shape *shape, uint32_t dim, uint32_t m);

primitiv_Shape *primitiv_Shape_resize_batch(const primitiv_Shape *shape, uint32_t batch);

void primitiv_Shape_update_dim(primitiv_Shape *shape, uint32_t dim, uint32_t m);

void primitiv_Shape_update_batch(primitiv_Shape *shape, uint32_t batch);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_SHAPE_H_
