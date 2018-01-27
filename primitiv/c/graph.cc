#include <primitiv/config.h>

#include <string>
#include <vector>

#include <primitiv/graph.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/graph.h>

using primitiv::Node;
using primitiv::Graph;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;
using primitiv::c::internal::to_c_ptr_from_value;

PRIMITIV_C_STATUS primitiv_Node_new(primitivNode_t **node) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  *node = to_c_ptr(new Node);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Node_clone(
    primitivNode_t *src, primitivNode_t **node) try {
  PRIMITIV_C_CHECK_NOT_NULL(src);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  *node = to_c_ptr(new Node(*to_cpp_ptr(src)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Node_delete(primitivNode_t *node) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  delete to_cpp_ptr(node);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Node_valid(
    const primitivNode_t *node, PRIMITIV_C_BOOL *valid) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(valid);
  *valid = to_cpp_ptr(node)->valid();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Node_graph(
    const primitivNode_t *node, primitivGraph_t **graph) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  *graph = to_c_ptr(&to_cpp_ptr(node)->graph());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Node_operator_id(
    const primitivNode_t *node, uint32_t *id) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(id);
  *id = to_cpp_ptr(node)->operator_id();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Node_value_id(
    const primitivNode_t *node, uint32_t *id) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(id);
  *id = to_cpp_ptr(node)->value_id();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Node_shape(
    const primitivNode_t *node, primitivShape_t **shape) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  *shape = to_c_ptr_from_value(to_cpp_ptr(node)->shape());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Node_device(
    const primitivNode_t *node, primitivDevice_t **device) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(device);
  *device = to_c_ptr(&to_cpp_ptr(node)->device());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Node_to_float(
    const primitivNode_t *node, float *value) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(value);
  *value = to_cpp_ptr(node)->to_float();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Node_to_array(
    const primitivNode_t *node, float *array, size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_vector_to_array(
      to_cpp_ptr(node)->to_vector(), array, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Node_argmax(
    const primitivNode_t *node, uint32_t dim, uint32_t *indices,
    size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_vector_to_array(
      to_cpp_ptr(node)->argmax(dim), indices, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Node_argmin(
    const primitivNode_t *node, uint32_t dim, uint32_t *indices,
    size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_vector_to_array(
      to_cpp_ptr(node)->argmin(dim), indices, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Node_backward(const primitivNode_t *node) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  to_cpp_ptr(node)->backward();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Graph_new(primitivGraph_t **graph) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  *graph = to_c_ptr(new Graph());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Graph_delete(primitivGraph_t *graph) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  delete to_cpp_ptr(graph);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Graph_get_default(primitivGraph_t **graph) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  *graph = to_c_ptr(&Graph::get_default());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Graph_set_default(primitivGraph_t *graph) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  Graph::set_default(*to_cpp_ptr(graph));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Graph_clear(primitivGraph_t *graph) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  to_cpp_ptr(graph)->clear();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Graph_forward(
    primitivGraph_t *graph, const primitivNode_t *node,
    const primitivTensor_t **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr(&to_cpp_ptr(graph)->forward(*to_cpp_ptr(node)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Graph_backward(
    primitivGraph_t *graph, const primitivNode_t *node) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  to_cpp_ptr(graph)->backward(*to_cpp_ptr(node));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Graph_get_shape(
    const primitivGraph_t *graph, const primitivNode_t *node,
    primitivShape_t **shape) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  *shape = to_c_ptr_from_value(
      to_cpp_ptr(graph)->get_shape(*to_cpp_ptr(node)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Graph_get_device(
    const primitivGraph_t *graph, const primitivNode_t *node,
    primitivDevice_t **device) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(device);
  *device = to_c_ptr(&to_cpp_ptr(graph)->get_device(*to_cpp_ptr(node)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Graph_dump(
    const primitivGraph_t *graph, const char *format, char *buffer,
    size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  PRIMITIV_C_CHECK_NOT_NULL(format);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_string_to_array(
      to_cpp_ptr(graph)->dump(format), buffer, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Graph_num_operators(
    const primitivGraph_t *graph, uint32_t *num) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  PRIMITIV_C_CHECK_NOT_NULL(num);
  *num = to_cpp_ptr(graph)->num_operators();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
