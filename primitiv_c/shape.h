#ifndef PRIMITIV_C_SHAPE_H_
#define PRIMITIV_C_SHAPE_H_

#include "primitiv_c/define.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct primitiv_Shape primitiv_Shape;

primitiv_Shape *primitiv_Shape_new();
primitiv_Shape *safe_primitiv_Shape_new(primitiv_Status *status);

primitiv_Shape *primitiv_Shape_new_with_dims(const uint32_t *dims, size_t n, uint32_t batch);
primitiv_Shape *safe_primitiv_Shape_new_with_dims(const uint32_t *dims, size_t n, uint32_t batch, primitiv_Status *status);

void primitiv_Shape_delete(primitiv_Shape *shape);
void safe_primitiv_Shape_delete(primitiv_Shape *shape, primitiv_Status *status);

uint32_t primitiv_Shape_op_getitem(const primitiv_Shape *shape, uint32_t i);
uint32_t safe_primitiv_Shape_op_getitem(const primitiv_Shape *shape, uint32_t i, primitiv_Status *status);

const uint32_t *primitiv_Shape_dims(const primitiv_Shape *shape);
const uint32_t *safe_primitiv_Shape_dims(const primitiv_Shape *shape, primitiv_Status *status);

uint32_t primitiv_Shape_depth(const primitiv_Shape *shape);
uint32_t safe_primitiv_Shape_depth(const primitiv_Shape *shape, primitiv_Status *status);

uint32_t primitiv_Shape_batch(const primitiv_Shape *shape);
uint32_t safe_primitiv_Shape_batch(const primitiv_Shape *shape, primitiv_Status *status);

uint32_t primitiv_Shape_volume(const primitiv_Shape *shape);
uint32_t safe_primitiv_Shape_volume(const primitiv_Shape *shape, primitiv_Status *status);

uint32_t primitiv_Shape_lower_volume(const primitiv_Shape *shape, uint32_t dim);
uint32_t safe_primitiv_Shape_lower_volume(const primitiv_Shape *shape, uint32_t dim, primitiv_Status *status);

uint32_t primitiv_Shape_size(const primitiv_Shape *shape);
uint32_t safe_primitiv_Shape_size(const primitiv_Shape *shape, primitiv_Status *status);

char *primitiv_Shape_to_string(const primitiv_Shape *shape);
char *safe_primitiv_Shape_to_string(const primitiv_Shape *shape, primitiv_Status *status);

bool primitiv_Shape_op_eq(const primitiv_Shape *shape, const primitiv_Shape *rhs);
bool safe_primitiv_Shape_op_eq(const primitiv_Shape *shape, const primitiv_Shape *rhs, primitiv_Status *status);

bool primitiv_Shape_op_ne(const primitiv_Shape *shape, const primitiv_Shape *rhs);
bool safe_primitiv_Shape_op_ne(const primitiv_Shape *shape, const primitiv_Shape *rhs, primitiv_Status *status);

bool primitiv_Shape_has_batch(const primitiv_Shape *shape);
bool safe_primitiv_Shape_has_batch(const primitiv_Shape *shape, primitiv_Status *status);

bool primitiv_Shape_has_compatible_batch(const primitiv_Shape *shape, const primitiv_Shape *rhs);
bool safe_primitiv_Shape_has_compatible_batch(const primitiv_Shape *shape, const primitiv_Shape *rhs, primitiv_Status *status);

bool primitiv_Shape_is_scalar(const primitiv_Shape *shape);
bool safe_primitiv_Shape_is_scalar(const primitiv_Shape *shape, primitiv_Status *status);

bool primitiv_Shape_is_row_vector(const primitiv_Shape *shape);
bool safe_primitiv_Shape_is_row_vector(const primitiv_Shape *shape, primitiv_Status *status);

bool primitiv_Shape_is_matrix(const primitiv_Shape *shape);
bool safe_primitiv_Shape_is_matrix(const primitiv_Shape *shape, primitiv_Status *status);

bool primitiv_Shape_has_same_dims(const primitiv_Shape *shape, const primitiv_Shape *rhs);
bool safe_primitiv_Shape_has_same_dims(const primitiv_Shape *shape, const primitiv_Shape *rhs, primitiv_Status *status);

bool primitiv_Shape_has_same_loo_dims(const primitiv_Shape *shape, const primitiv_Shape *rhs, uint32_t dim);
bool safe_primitiv_Shape_has_same_loo_dims(const primitiv_Shape *shape, const primitiv_Shape *rhs, uint32_t dim, primitiv_Status *status);

primitiv_Shape *primitiv_Shape_resize_dim(const primitiv_Shape *shape, uint32_t dim, uint32_t m);
primitiv_Shape *safe_primitiv_Shape_resize_dim(const primitiv_Shape *shape, uint32_t dim, uint32_t m, primitiv_Status *status);

primitiv_Shape *primitiv_Shape_resize_batch(const primitiv_Shape *shape, uint32_t batch);
primitiv_Shape *safe_primitiv_Shape_resize_batch(const primitiv_Shape *shape, uint32_t batch, primitiv_Status *status);

void primitiv_Shape_update_dim(primitiv_Shape *shape, uint32_t dim, uint32_t m);
void safe_primitiv_Shape_update_dim(primitiv_Shape *shape, uint32_t dim, uint32_t m, primitiv_Status *status);

void primitiv_Shape_update_batch(primitiv_Shape *shape, uint32_t batch);
void safe_primitiv_Shape_update_batch(primitiv_Shape *shape, uint32_t batch, primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_SHAPE_H_
