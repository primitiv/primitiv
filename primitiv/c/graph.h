/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_GRAPH_H_
#define PRIMITIV_C_GRAPH_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>
#include <primitiv/c/shape.h>
#include <primitiv/c/status.h>
#include <primitiv/c/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct primitiv_Node primitiv_Node;

typedef struct primitiv_Graph primitiv_Graph;

CAPI extern primitiv_Node *primitiv_Node_new();
CAPI extern primitiv_Node *safe_primitiv_Node_new(primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_Node_new_with_movement(primitiv_Node *node);
CAPI extern primitiv_Node *safe_primitiv_Node_new_with_movement(
    primitiv_Node *node, primitiv_Status *status);

CAPI extern void primitiv_Node_delete(primitiv_Node *node);
CAPI extern void safe_primitiv_Node_delete(primitiv_Node *node,
                                           primitiv_Status *status);

CAPI extern bool primitiv_Node_valid(const primitiv_Node *node);
CAPI extern bool safe_primitiv_Node_valid(const primitiv_Node *node,
                                          primitiv_Status *status);

CAPI extern primitiv_Graph *primitiv_Node_graph(const primitiv_Node *node);
CAPI extern primitiv_Graph *safe_primitiv_Node_graph(const primitiv_Node *node,
                                                     primitiv_Status *status);

CAPI extern uint32_t primitiv_Node_operator_id(const primitiv_Node *node);
CAPI extern uint32_t safe_primitiv_Node_operator_id(const primitiv_Node *node,
                                                    primitiv_Status *status);

CAPI extern uint32_t primitiv_Node_value_id(const primitiv_Node *node);
CAPI extern uint32_t safe_primitiv_Node_value_id(const primitiv_Node *node,
                                                 primitiv_Status *status);

CAPI extern primitiv_Shape *primitiv_Node_shape(const primitiv_Node *node);
CAPI extern primitiv_Shape *safe_primitiv_Node_shape(const primitiv_Node *node,
                                                     primitiv_Status *status);

CAPI extern primitiv_Device *primitiv_Node_device(const primitiv_Node *node);
CAPI extern primitiv_Device *safe_primitiv_Node_device(
    const primitiv_Node *node, primitiv_Status *status);

CAPI extern float primitiv_Node_to_float(const primitiv_Node *node);
CAPI extern float safe_primitiv_Node_to_float(const primitiv_Node *node,
                                              primitiv_Status *status);

CAPI extern void primitiv_Node_to_array(const primitiv_Node *node,
                                        float *array);
CAPI extern void safe_primitiv_Node_to_array(const primitiv_Node *node,
                                             float *array,
                                             primitiv_Status *status);

CAPI extern uint32_t *primitiv_Node_argmax(const primitiv_Node *node,
                                           uint32_t dim);
CAPI extern uint32_t *safe_primitiv_Node_argmax(const primitiv_Node *node,
                                                uint32_t dim,
                                                primitiv_Status *status);

CAPI extern uint32_t *primitiv_Node_argmin(const primitiv_Node *node,
                                           uint32_t dim);
CAPI extern uint32_t *safe_primitiv_Node_argmin(const primitiv_Node *node,
                                                uint32_t dim,
                                                primitiv_Status *status);

CAPI extern void primitiv_Node_backward(const primitiv_Node *node);
CAPI extern void safe_primitiv_Node_backward(const primitiv_Node *node,
                                             primitiv_Status *status);

CAPI extern primitiv_Graph *primitiv_Graph_new();
CAPI extern primitiv_Graph *safe_primitiv_Graph_new(primitiv_Status *status);

CAPI extern void primitiv_Graph_delete(primitiv_Graph *graph);
CAPI extern void safe_primitiv_Graph_delete(primitiv_Graph *graph,
                                            primitiv_Status *status);

CAPI extern primitiv_Graph *primitiv_Graph_get_default();
CAPI extern primitiv_Graph *safe_primitiv_Graph_get_default(
    primitiv_Status *status);

CAPI extern void primitiv_Graph_set_default(primitiv_Graph *graph);
CAPI extern void safe_primitiv_Graph_set_default(primitiv_Graph *graph,
                                                 primitiv_Status *status);

CAPI extern void primitiv_Graph_clear(primitiv_Graph *graph);
CAPI extern void safe_primitiv_Graph_clear(primitiv_Graph *graph,
                                           primitiv_Status *status);

CAPI extern const primitiv_Tensor *primitiv_Graph_forward(
    primitiv_Graph *graph, const primitiv_Node *node);
CAPI extern const primitiv_Tensor *safe_primitiv_Graph_forward(
    primitiv_Graph *graph, const primitiv_Node *node, primitiv_Status *status);

CAPI extern void primitiv_Graph_backward(primitiv_Graph *graph,
                                         const primitiv_Node *node);
CAPI extern void safe_primitiv_Graph_backward(primitiv_Graph *graph,
                                              const primitiv_Node *node,
                                              primitiv_Status *status);

CAPI extern primitiv_Shape *primitiv_Graph_get_shape(
    const primitiv_Graph *graph, const primitiv_Node *node);
CAPI extern primitiv_Shape *safe_primitiv_Graph_get_shape(
    const primitiv_Graph *graph,
    const primitiv_Node *node,
    primitiv_Status *status);

CAPI extern primitiv_Device *primitiv_Graph_get_device(
    const primitiv_Graph *graph, const primitiv_Node *node);
CAPI extern primitiv_Device *safe_primitiv_Graph_get_device(
    const primitiv_Graph *graph,
    const primitiv_Node *node,
    primitiv_Status *status);

CAPI extern char *primitiv_Graph_dump(const primitiv_Graph *graph,
                                      const char *format);
CAPI extern char *safe_primitiv_Graph_dump(const primitiv_Graph *graph,
                                           const char *format,
                                           primitiv_Status *status);

CAPI extern uint32_t primitiv_Graph_num_operators(const primitiv_Graph *graph);
CAPI extern uint32_t safe_primitiv_Graph_num_operators(
    const primitiv_Graph *graph, primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_GRAPH_H_
