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
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;
using primitiv::c::internal::to_c_ptr_from_value;

extern "C" {

primitiv_Status primitiv_Node_new(primitiv_Node **node) try {
  *node = to_c_ptr(new Node);
  return ::primitiv_Status::PRIMITIV_OK;
} HANDLE_EXCEPTION

primitiv_Status primitiv_Node_clone(
    primitiv_Node *src, primitiv_Node **node) try {
  *node = to_c_ptr(new Node(*to_cpp_ptr(src)));
  return ::primitiv_Status::PRIMITIV_OK;
} HANDLE_EXCEPTION

void primitiv_Node_delete(primitiv_Node *node) {
  delete to_cpp_ptr(node);
}

bool primitiv_Node_valid(const primitiv_Node *node) {
  return to_cpp_ptr(node)->valid();
}

primitiv_Status primitiv_Node_graph(const primitiv_Node *node,
                                    primitiv_Graph **graph) {
  try {
    *graph = to_c_ptr(&to_cpp_ptr(node)->graph());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Node_operator_id(const primitiv_Node *node,
                                          uint32_t *id) {
  try {
    *id = to_cpp_ptr(node)->operator_id();
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Node_value_id(const primitiv_Node *node,
                                       uint32_t *id) {
  try {
    *id = to_cpp_ptr(node)->value_id();
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Node_shape(const primitiv_Node *node,
                                    primitiv_Shape **shape) {
  try {
    *shape = to_c_ptr_from_value(to_cpp_ptr(node)->shape());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Node_device(const primitiv_Node *node,
                                     primitiv_Device **device) {
  try {
    *device = to_c_ptr(&to_cpp_ptr(node)->device());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Node_to_float(const primitiv_Node *node,
                                       float *value) {
  try {
    *value = to_cpp_ptr(node)->to_float();
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Node_to_array(const primitiv_Node *node,
                                       float *array) {
  try {
    std::vector<float> v = to_cpp_ptr(node)->to_vector();
    std::copy(v.begin(), v.end(), array);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Node_argmax(
    const primitiv_Node *node, uint32_t dim, uint32_t *indices) {
  try {
    std::vector<uint32_t> v = to_cpp_ptr(node)->argmax(dim);
    std::copy(v.begin(), v.end(), indices);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Node_argmin(
    const primitiv_Node *node, uint32_t dim, uint32_t *indices) {
  try {
    std::vector<uint32_t> v = to_cpp_ptr(node)->argmin(dim);
    std::copy(v.begin(), v.end(), indices);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Node_backward(const primitiv_Node *node) {
  try {
    to_cpp_ptr(node)->backward();
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Graph_new(primitiv_Graph **graph) try {
  *graph = to_c_ptr(new Graph());
  return ::primitiv_Status::PRIMITIV_OK;
} HANDLE_EXCEPTION

void primitiv_Graph_delete(primitiv_Graph *graph) {
  delete to_cpp_ptr(graph);
}

primitiv_Status primitiv_Graph_get_default(primitiv_Graph **graph) {
  try {
    *graph = to_c_ptr(&Graph::get_default());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

void primitiv_Graph_set_default(primitiv_Graph *graph) {
  Graph::set_default(*to_cpp_ptr(graph));
}

void primitiv_Graph_clear(primitiv_Graph *graph) {
  to_cpp_ptr(graph)->clear();
}

primitiv_Status primitiv_Graph_forward(primitiv_Graph *graph,
                                       const primitiv_Node *node,
                                       const primitiv_Tensor **tensor) {
  try {
    *tensor = to_c_ptr(&to_cpp_ptr(graph)->forward(*to_cpp_ptr(node)));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Graph_backward(primitiv_Graph *graph,
                                        const primitiv_Node *node) {
  try {
    to_cpp_ptr(graph)->backward(*to_cpp_ptr(node));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Graph_get_shape(const primitiv_Graph *graph,
                                         const primitiv_Node *node,
                                         primitiv_Shape **shape) {
  try {
    *shape = to_c_ptr_from_value(
        to_cpp_ptr(graph)->get_shape(*to_cpp_ptr(node)));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Graph_get_device(const primitiv_Graph *graph,
                                          const primitiv_Node *node,
                                          primitiv_Device **device) {
  try {
    *device = to_c_ptr(&to_cpp_ptr(graph)->get_device(*to_cpp_ptr(node)));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Graph_dump(
    const primitiv_Graph *graph, const char *format, char *string,
    size_t *length) {
  try {
    std::string str = to_cpp_ptr(graph)->dump(format);
    *length = str.length();
    if (string) {
      std::strcpy(string, str.c_str());
    }
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

uint32_t primitiv_Graph_num_operators(const primitiv_Graph *graph) {
  return to_cpp_ptr(graph)->num_operators();
}

}  // end extern "C"
