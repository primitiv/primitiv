#ifndef PRIMITIV_C_GRAPH_H_
#define PRIMITIV_C_GRAPH_H_

#include "primitiv_c/define.h"

#include "primitiv_c/device.h"
#include "primitiv_c/shape.h"
#include "primitiv_c/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct primitiv_Node primitiv_Node;

typedef struct primitiv_Graph primitiv_Graph;

primitiv_Node *primitiv_Node_new();

primitiv_Node *primitiv_Node_new_with_movement(primitiv_Node *node);

void primitiv_Node_delete(primitiv_Node *node);

bool primitiv_Node_valid(const primitiv_Node *node);

primitiv_Graph *primitiv_Node_graph(const primitiv_Node *node);

uint32_t primitiv_Node_function_id(const primitiv_Node *node);

uint32_t primitiv_Node_value_id(const primitiv_Node *node);

primitiv_Shape *primitiv_Node_shape(const primitiv_Node *node);

primitiv_Device *primitiv_Node_device(const primitiv_Node *node);

float primitiv_Node_to_float(const primitiv_Node *node);

float *primitiv_Node_to_array(const primitiv_Node *node);

uint32_t *primitiv_Node_argmax(const primitiv_Node *node, uint32_t dim);

uint32_t *primitiv_Node_argmin(const primitiv_Node *node, uint32_t dim);

void primitiv_Node_backward(const primitiv_Node *node);

primitiv_Graph *primitiv_Graph_new();

void primitiv_Graph_delete(primitiv_Graph *graph);

primitiv_Graph *primitiv_Graph_get_default();

void primitiv_Graph_set_default(primitiv_Graph *graph);

void primitiv_Graph_clear(primitiv_Graph *graph);

const primitiv_Tensor *primitiv_Graph_forward(primitiv_Graph *graph, const primitiv_Node *node);

void primitiv_Graph_backward(primitiv_Graph *graph, const primitiv_Node *node);

primitiv_Shape *primitiv_Graph_get_shape(const primitiv_Graph *graph, const primitiv_Node *node);

primitiv_Device *primitiv_Graph_get_device(const primitiv_Graph *graph, const primitiv_Node *node);

char *primitiv_Graph_dump(const primitiv_Graph *graph, const char *format);

uint32_t primitiv_Graph_num_functions(const primitiv_Graph *graph);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_GRAPH_H_
