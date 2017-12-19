/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <config.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <primitiv/graph.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/graph.h>

using primitiv::Node;
using primitiv::Graph;
using primitiv_c::internal::to_c;
using primitiv_c::internal::to_cc;
using primitiv_c::internal::to_c_from_value;

extern "C" {

primitiv_Node *primitiv_Node_new() {
  return to_c(new Node);
}

primitiv_Status primitiv_Node_new_from_node(primitiv_Node **node,
                                            primitiv_Node *src) {
  try {
    *node = to_c(new Node(std::move(*to_cc(src))));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

void primitiv_Node_delete(primitiv_Node *node) {
  delete to_cc(node);
}

bool primitiv_Node_valid(const primitiv_Node *node) {
  return to_cc(node)->valid();
}

primitiv_Status primitiv_Node_graph(const primitiv_Node *node,
                                    primitiv_Graph **graph) {
  try {
    *graph = to_c(&to_cc(node)->graph());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Node_operator_id(const primitiv_Node *node,
                                          uint32_t *id) {
  try {
    *id = to_cc(node)->operator_id();
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Node_value_id(const primitiv_Node *node,
                                       uint32_t *id) {
  try {
    *id = to_cc(node)->value_id();
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Node_shape(const primitiv_Node *node,
                                    primitiv_Shape **shape) {
  try {
    *shape = to_c_from_value(to_cc(node)->shape());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Node_device(const primitiv_Node *node,
                                     primitiv_Device **device) {
  try {
    *device = to_c(&to_cc(node)->device());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Node_to_float(const primitiv_Node *node,
                                       float *value) {
  try {
    *value = to_cc(node)->to_float();
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Node_to_array(const primitiv_Node *node,
                                       float *array) {
  try {
    std::vector<float> v = to_cc(node)->to_vector();
    std::copy(v.begin(), v.end(), array);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Node_argmax(
    const primitiv_Node *node, uint32_t dim, uint32_t *indices) {
  try {
    std::vector<uint32_t> v = to_cc(node)->argmax(dim);
    std::copy(v.begin(), v.end(), indices);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Node_argmin(
    const primitiv_Node *node, uint32_t dim, uint32_t *indices) {
  try {
    std::vector<uint32_t> v = to_cc(node)->argmin(dim);
    std::copy(v.begin(), v.end(), indices);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Node_backward(const primitiv_Node *node) {
  try {
    to_cc(node)->backward();
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Graph *primitiv_Graph_new() {
  return to_c(new Graph());
}

void primitiv_Graph_delete(primitiv_Graph *graph) {
  delete to_cc(graph);
}

primitiv_Status primitiv_Graph_get_default(primitiv_Graph **graph) {
  try {
    *graph = to_c(&Graph::get_default());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

void primitiv_Graph_set_default(primitiv_Graph *graph) {
  Graph::set_default(*to_cc(graph));
}

void primitiv_Graph_clear(primitiv_Graph *graph) {
  to_cc(graph)->clear();
}

primitiv_Status primitiv_Graph_forward(primitiv_Graph *graph,
                                       const primitiv_Node *node,
                                       const primitiv_Tensor **tensor) {
  try {
    *tensor = to_c(&to_cc(graph)->forward(*to_cc(node)));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Graph_backward(primitiv_Graph *graph,
                                        const primitiv_Node *node) {
  try {
    to_cc(graph)->backward(*to_cc(node));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Graph_get_shape(const primitiv_Graph *graph,
                                         const primitiv_Node *node,
                                         primitiv_Shape **shape) {
  try {
    *shape = to_c_from_value(to_cc(graph)->get_shape(*to_cc(node)));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Graph_get_device(const primitiv_Graph *graph,
                                          const primitiv_Node *node,
                                          primitiv_Device **device) {
  try {
    *device = to_c(&to_cc(graph)->get_device(*to_cc(node)));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Graph_dump(
    const primitiv_Graph *graph, const char *format, char *string) {
  try {
    std::string str = to_cc(graph)->dump(format);
    uint64_t len = str.length();
    std::strncpy(string, str.c_str(), len);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

uint32_t primitiv_Graph_num_operators(const primitiv_Graph *graph) {
  return to_cc(graph)->num_operators();
}

}  // end extern "C"
