#include "primitiv_c/internal.h"
#include "primitiv_c/graph.h"

#include <string>
#include <utility>

#include <primitiv/graph.h>

using primitiv::Node;
using primitiv::Graph;

extern "C" {

primitiv_Node *primitiv_Node_new() {
  return to_c(new Node);
}

primitiv_Node *primitiv_Node_new_with_movement(primitiv_Node *node) {
  return to_c(new Node(std::move(*to_cc(node))));
}

void primitiv_Node_delete(primitiv_Node *node) {
  delete to_cc(node);
}

bool primitiv_Node_valid(const primitiv_Node *node) {
  return to_cc(node)->valid();
}

primitiv_Graph *primitiv_Node_graph(const primitiv_Node *node) {
  return to_c(&to_cc(node)->graph());
}

uint32_t primitiv_Node_function_id(const primitiv_Node *node) {
  return to_cc(node)->function_id();
}

uint32_t primitiv_Node_value_id(const primitiv_Node *node) {
  return to_cc(node)->value_id();
}

primitiv_Shape *primitiv_Node_shape(const primitiv_Node *node) {
  return to_c_from_value(to_cc(node)->shape());
}

primitiv_Device *primitiv_Node_device(const primitiv_Node *node) {
  return to_c(&to_cc(node)->device());
}

float primitiv_Node_to_float(const primitiv_Node *node) {
  return to_cc(node)->to_float();
}

float *primitiv_Node_to_array(const primitiv_Node *node) {
  return &(to_cc(node)->to_vector())[0];
}

uint32_t *primitiv_Node_argmax(const primitiv_Node *node, uint32_t dim) {
  return &(to_cc(node)->argmax(dim))[0];
}

uint32_t *primitiv_Node_argmin(const primitiv_Node *node, uint32_t dim) {
  return &(to_cc(node)->argmin(dim))[0];
}

void primitiv_Node_backward(const primitiv_Node *node) {
  to_cc(node)->backward();
}

primitiv_Graph *primitiv_Graph_new() {
  return to_c(new Graph());
}

void primitiv_Graph_delete(primitiv_Graph *graph) {
  delete to_cc(graph);
}

primitiv_Graph *primitiv_Graph_get_default() {
  return to_c(&Graph::get_default());
}

void primitiv_Graph_set_default(primitiv_Graph *graph) {
  Graph::set_default(*to_cc(graph));
}

void primitiv_Graph_clear(primitiv_Graph *graph) {
  to_cc(graph)->clear();
}

const primitiv_Tensor *primitiv_Graph_forward(primitiv_Graph *graph, const primitiv_Node *node) {
  return to_c(&to_cc(graph)->forward(*to_cc(node)));
}

void primitiv_Graph_backward(primitiv_Graph *graph, const primitiv_Node *node) {
  to_cc(graph)->backward(*to_cc(node));
}

primitiv_Shape *primitiv_Graph_get_shape(const primitiv_Graph *graph, const primitiv_Node *node) {
  return to_c_from_value(to_cc(graph)->get_shape(*to_cc(node)));
}

primitiv_Device *primitiv_Graph_get_device(const primitiv_Graph *graph, const primitiv_Node *node) {
  return to_c(&to_cc(graph)->get_device(*to_cc(node)));
}

char *primitiv_Graph_dump(const primitiv_Graph *graph, const char *format) {
  std::string str = to_cc(graph)->dump(format);
  unsigned long len = str.length();
  char *c = new char[len + 1];
  std::strncpy(c, str.c_str(), len);
  return c;
}

uint32_t primitiv_Graph_num_functions(const primitiv_Graph *graph) {
  return to_cc(graph)->num_functions();
}

}  // end extern "C"
