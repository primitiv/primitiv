#ifndef PRIMITIV_C_GRAPH_H_
#define PRIMITIV_C_GRAPH_H_

#include "primitiv_c/define.h"
#include "primitiv_c/device.h"
#include "primitiv_c/shape.h"
#include "primitiv_c/status.h"
#include "primitiv_c/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct primitiv_Node primitiv_Node;

typedef struct primitiv_Graph primitiv_Graph;

primitiv_Node *primitiv_Node_new();
primitiv_Node *safe_primitiv_Node_new(primitiv_Status *status);

primitiv_Node *primitiv_Node_new_with_movement(primitiv_Node *node);
primitiv_Node *safe_primitiv_Node_new_with_movement(primitiv_Node *node, primitiv_Status *status);

void primitiv_Node_delete(primitiv_Node *node);
void safe_primitiv_Node_delete(primitiv_Node *node, primitiv_Status *status);

bool primitiv_Node_valid(const primitiv_Node *node);
bool safe_primitiv_Node_valid(const primitiv_Node *node, primitiv_Status *status);

primitiv_Graph *primitiv_Node_graph(const primitiv_Node *node);
primitiv_Graph *safe_primitiv_Node_graph(const primitiv_Node *node, primitiv_Status *status);

uint32_t primitiv_Node_operator_id(const primitiv_Node *node);
uint32_t safe_primitiv_Node_operator_id(const primitiv_Node *node, primitiv_Status *status);

uint32_t primitiv_Node_value_id(const primitiv_Node *node);
uint32_t safe_primitiv_Node_value_id(const primitiv_Node *node, primitiv_Status *status);

primitiv_Shape *primitiv_Node_shape(const primitiv_Node *node);
primitiv_Shape *safe_primitiv_Node_shape(const primitiv_Node *node, primitiv_Status *status);

primitiv_Device *primitiv_Node_device(const primitiv_Node *node);
primitiv_Device *safe_primitiv_Node_device(const primitiv_Node *node, primitiv_Status *status);

float primitiv_Node_to_float(const primitiv_Node *node);
float safe_primitiv_Node_to_float(const primitiv_Node *node, primitiv_Status *status);

float *primitiv_Node_to_array(const primitiv_Node *node);
float *safe_primitiv_Node_to_array(const primitiv_Node *node, primitiv_Status *status);

uint32_t *primitiv_Node_argmax(const primitiv_Node *node, uint32_t dim);
uint32_t *safe_primitiv_Node_argmax(const primitiv_Node *node, uint32_t dim, primitiv_Status *status);

uint32_t *primitiv_Node_argmin(const primitiv_Node *node, uint32_t dim);
uint32_t *safe_primitiv_Node_argmin(const primitiv_Node *node, uint32_t dim, primitiv_Status *status);

void primitiv_Node_backward(const primitiv_Node *node);
void safe_primitiv_Node_backward(const primitiv_Node *node, primitiv_Status *status);

primitiv_Graph *primitiv_Graph_new();
primitiv_Graph *safe_primitiv_Graph_new(primitiv_Status *status);

void primitiv_Graph_delete(primitiv_Graph *graph);
void safe_primitiv_Graph_delete(primitiv_Graph *graph, primitiv_Status *status);

primitiv_Graph *primitiv_Graph_get_default();
primitiv_Graph *safe_primitiv_Graph_get_default(primitiv_Status *status);

void primitiv_Graph_set_default(primitiv_Graph *graph);
void safe_primitiv_Graph_set_default(primitiv_Graph *graph, primitiv_Status *status);

void primitiv_Graph_clear(primitiv_Graph *graph);
void safe_primitiv_Graph_clear(primitiv_Graph *graph, primitiv_Status *status);

const primitiv_Tensor *primitiv_Graph_forward(primitiv_Graph *graph, const primitiv_Node *node);
const primitiv_Tensor *safe_primitiv_Graph_forward(primitiv_Graph *graph, const primitiv_Node *node, primitiv_Status *status);

void primitiv_Graph_backward(primitiv_Graph *graph, const primitiv_Node *node);
void safe_primitiv_Graph_backward(primitiv_Graph *graph, const primitiv_Node *node, primitiv_Status *status);

primitiv_Shape *primitiv_Graph_get_shape(const primitiv_Graph *graph, const primitiv_Node *node);
primitiv_Shape *safe_primitiv_Graph_get_shape(const primitiv_Graph *graph, const primitiv_Node *node, primitiv_Status *status);

primitiv_Device *primitiv_Graph_get_device(const primitiv_Graph *graph, const primitiv_Node *node);
primitiv_Device *safe_primitiv_Graph_get_device(const primitiv_Graph *graph, const primitiv_Node *node, primitiv_Status *status);

char *primitiv_Graph_dump(const primitiv_Graph *graph, const char *format);
char *safe_primitiv_Graph_dump(const primitiv_Graph *graph, const char *format, primitiv_Status *status);

uint32_t primitiv_Graph_num_operators(const primitiv_Graph *graph);
uint32_t safe_primitiv_Graph_num_operators(const primitiv_Graph *graph, primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_GRAPH_H_
