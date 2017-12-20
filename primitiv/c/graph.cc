/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <primitiv/graph.h>

#include <primitiv/c/internal.h>
#include <primitiv/c/graph.h>

using primitiv::Node;
using primitiv::Graph;

extern "C" {

primitiv_Node *primitiv_Node_new() {
  return to_c(new Node);
}
primitiv_Node *safe_primitiv_Node_new(primitiv_Status *status) {
  SAFE_RETURN(primitiv_Node_new(), status, nullptr);
}

primitiv_Node *primitiv_Node_new_with_movement(primitiv_Node *node) {
  return to_c(new Node(std::move(*to_cc(node))));
}
primitiv_Node *safe_primitiv_Node_new_with_movement(primitiv_Node *node,
                                                    primitiv_Status *status) {
  SAFE_RETURN(primitiv_Node_new_with_movement(node), status, nullptr);
}

void primitiv_Node_delete(primitiv_Node *node) {
  delete to_cc(node);
}
void safe_primitiv_Node_delete(primitiv_Node *node, primitiv_Status *status) {
  SAFE_EXPR(primitiv_Node_delete(node), status);
}

bool primitiv_Node_valid(const primitiv_Node *node) {
  return to_cc(node)->valid();
}
bool safe_primitiv_Node_valid(const primitiv_Node *node,
                              primitiv_Status *status) {
  SAFE_RETURN(primitiv_Node_valid(node), status, false);
}

primitiv_Graph *primitiv_Node_graph(const primitiv_Node *node) {
  return to_c(&to_cc(node)->graph());
}
primitiv_Graph *safe_primitiv_Node_graph(const primitiv_Node *node,
                                         primitiv_Status *status) {
  SAFE_RETURN(primitiv_Node_graph(node), status, nullptr);
}

uint32_t primitiv_Node_operator_id(const primitiv_Node *node) {
  return to_cc(node)->operator_id();
}
uint32_t safe_primitiv_Node_operator_id(const primitiv_Node *node,
                                        primitiv_Status *status) {
  SAFE_RETURN(primitiv_Node_operator_id(node), status, 0);
}

uint32_t primitiv_Node_value_id(const primitiv_Node *node) {
  return to_cc(node)->value_id();
}
uint32_t safe_primitiv_Node_value_id(const primitiv_Node *node,
                                     primitiv_Status *status) {
  SAFE_RETURN(primitiv_Node_value_id(node), status, 0);
}

primitiv_Shape *primitiv_Node_shape(const primitiv_Node *node) {
  return to_c_from_value(to_cc(node)->shape());
}
primitiv_Shape *safe_primitiv_Node_shape(const primitiv_Node *node,
                                         primitiv_Status *status) {
  SAFE_RETURN(primitiv_Node_shape(node), status, nullptr);
}

primitiv_Device *primitiv_Node_device(const primitiv_Node *node) {
  return to_c(&to_cc(node)->device());
}
primitiv_Device *safe_primitiv_Node_device(const primitiv_Node *node,
                                           primitiv_Status *status) {
  SAFE_RETURN(primitiv_Node_device(node), status, nullptr);
}

float primitiv_Node_to_float(const primitiv_Node *node) {
  return to_cc(node)->to_float();
}
float safe_primitiv_Node_to_float(const primitiv_Node *node,
                                  primitiv_Status *status) {
  SAFE_RETURN(primitiv_Node_to_float(node), status, 0.0);
}

void primitiv_Node_to_array(const primitiv_Node *node, float *array) {
  std::vector<float> v = to_cc(node)->to_vector();
  std::copy(v.begin(), v.end(), array);
}
void safe_primitiv_Node_to_array(const primitiv_Node *node,
                                 float *array,
                                 primitiv_Status *status) {
  SAFE_EXPR(primitiv_Node_to_array(node, array), status);
}

uint32_t *primitiv_Node_argmax(const primitiv_Node *node, uint32_t dim) {
  return &(to_cc(node)->argmax(dim))[0];
}
uint32_t *safe_primitiv_Node_argmax(const primitiv_Node *node,
                                    uint32_t dim,
                                    primitiv_Status *status) {
  SAFE_RETURN(primitiv_Node_argmax(node, dim), status, nullptr);
}

uint32_t *primitiv_Node_argmin(const primitiv_Node *node, uint32_t dim) {
  return &(to_cc(node)->argmin(dim))[0];
}
uint32_t *safe_primitiv_Node_argmin(const primitiv_Node *node,
                                    uint32_t dim,
                                    primitiv_Status *status) {
  SAFE_RETURN(primitiv_Node_argmin(node, dim), status, nullptr);
}

void primitiv_Node_backward(const primitiv_Node *node) {
  to_cc(node)->backward();
}
void safe_primitiv_Node_backward(const primitiv_Node *node,
                                 primitiv_Status *status) {
  SAFE_EXPR(primitiv_Node_backward(node), status);
}

primitiv_Graph *primitiv_Graph_new() {
  return to_c(new Graph());
}
primitiv_Graph *safe_primitiv_Graph_new(primitiv_Status *status) {
  SAFE_RETURN(primitiv_Graph_new(), status, nullptr);
}

void primitiv_Graph_delete(primitiv_Graph *graph) {
  delete to_cc(graph);
}
void safe_primitiv_Graph_delete(primitiv_Graph *graph,
                                primitiv_Status *status) {
  SAFE_EXPR(primitiv_Graph_delete(graph), status);
}

primitiv_Graph *primitiv_Graph_get_default() {
  return to_c(&Graph::get_default());
}
primitiv_Graph *safe_primitiv_Graph_get_default(primitiv_Status *status) {
  SAFE_RETURN(primitiv_Graph_get_default(), status, nullptr);
}

void primitiv_Graph_set_default(primitiv_Graph *graph) {
  Graph::set_default(*to_cc(graph));
}
void safe_primitiv_Graph_set_default(primitiv_Graph *graph,
                                     primitiv_Status *status) {
  SAFE_EXPR(primitiv_Graph_set_default(graph), status);
}

void primitiv_Graph_clear(primitiv_Graph *graph) {
  to_cc(graph)->clear();
}
void safe_primitiv_Graph_clear(primitiv_Graph *graph, primitiv_Status *status) {
  SAFE_EXPR(primitiv_Graph_clear(graph), status);
}

const primitiv_Tensor *primitiv_Graph_forward(primitiv_Graph *graph,
                                              const primitiv_Node *node) {
  return to_c(&to_cc(graph)->forward(*to_cc(node)));
}
const primitiv_Tensor *safe_primitiv_Graph_forward(primitiv_Graph *graph,
                                                   const primitiv_Node *node,
                                                   primitiv_Status *status) {
  SAFE_RETURN(primitiv_Graph_forward(graph, node), status, nullptr);
}

void primitiv_Graph_backward(primitiv_Graph *graph, const primitiv_Node *node) {
  to_cc(graph)->backward(*to_cc(node));
}
void safe_primitiv_Graph_backward(primitiv_Graph *graph,
                                  const primitiv_Node *node,
                                  primitiv_Status *status) {
  SAFE_EXPR(primitiv_Graph_backward(graph, node), status);
}

primitiv_Shape *primitiv_Graph_get_shape(const primitiv_Graph *graph,
                                         const primitiv_Node *node) {
  return to_c_from_value(to_cc(graph)->get_shape(*to_cc(node)));
}
primitiv_Shape *safe_primitiv_Graph_get_shape(const primitiv_Graph *graph,
                                              const primitiv_Node *node,
                                              primitiv_Status *status) {
  SAFE_RETURN(primitiv_Graph_get_shape(graph, node), status, nullptr);
}

primitiv_Device *primitiv_Graph_get_device(const primitiv_Graph *graph,
                                           const primitiv_Node *node) {
  return to_c(&to_cc(graph)->get_device(*to_cc(node)));
}
primitiv_Device *safe_primitiv_Graph_get_device(const primitiv_Graph *graph,
                                                const primitiv_Node *node,
                                                primitiv_Status *status) {
  SAFE_RETURN(primitiv_Graph_get_device(graph, node), status, nullptr);
}

char *primitiv_Graph_dump(const primitiv_Graph *graph, const char *format) {
  std::string str = to_cc(graph)->dump(format);
  uint64_t len = str.length();
  auto *c = new char[len + 1];
  std::strncpy(c, str.c_str(), len);
  return c;
}
char *safe_primitiv_Graph_dump(const primitiv_Graph *graph,
                               const char *format,
                               primitiv_Status *status) {
  SAFE_RETURN(primitiv_Graph_dump(graph, format), status, nullptr);
}

uint32_t primitiv_Graph_num_operators(const primitiv_Graph *graph) {
  return to_cc(graph)->num_operators();
}
uint32_t safe_primitiv_Graph_num_operators(const primitiv_Graph *graph,
                                           primitiv_Status *status) {
  SAFE_RETURN(primitiv_Graph_num_operators(graph), status, 0);
}

}  // end extern "C"
