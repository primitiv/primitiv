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

PRIMITIV_C_STATUS primitivCreateNode(primitivNode_t **node) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  *node = to_c_ptr(new Node);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCloneNode(
    primitivNode_t *src, primitivNode_t **node) try {
  PRIMITIV_C_CHECK_NOT_NULL(src);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  *node = to_c_ptr(new Node(*to_cpp_ptr(src)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivDeleteNode(primitivNode_t *node) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  delete to_cpp_ptr(node);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivIsValidNode(
    const primitivNode_t *node, PRIMITIV_C_BOOL *valid) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(valid);
  *valid = to_cpp_ptr(node)->valid();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetGraphFromNode(
    const primitivNode_t *node, primitivGraph_t **graph) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  *graph = to_c_ptr(&to_cpp_ptr(node)->graph());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetNodeOperatorId(
    const primitivNode_t *node, uint32_t *id) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(id);
  *id = to_cpp_ptr(node)->operator_id();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetNodeValueId(
    const primitivNode_t *node, uint32_t *id) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(id);
  *id = to_cpp_ptr(node)->value_id();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetNodeShape(
    const primitivNode_t *node, primitivShape_t **shape) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  *shape = to_c_ptr_from_value(to_cpp_ptr(node)->shape());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetDeviceFromNode(
    const primitivNode_t *node, primitivDevice_t **device) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(device);
  *device = to_c_ptr(&to_cpp_ptr(node)->device());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivEvaluateNodeAsFloat(
    const primitivNode_t *node, float *value) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(value);
  *value = to_cpp_ptr(node)->to_float();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivEvaluateNodeAsArray(
    const primitivNode_t *node, float *array, size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_vector_to_array(
      to_cpp_ptr(node)->to_vector(), array, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetNodeArgmax(
    const primitivNode_t *node, uint32_t dim, uint32_t *indices,
    size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_vector_to_array(
      to_cpp_ptr(node)->argmax(dim), indices, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetNodeArgmin(
    const primitivNode_t *node, uint32_t dim, uint32_t *indices,
    size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_vector_to_array(
      to_cpp_ptr(node)->argmin(dim), indices, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivExecuteNodeBackward(const primitivNode_t *node) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  to_cpp_ptr(node)->backward();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateGraph(primitivGraph_t **graph) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  *graph = to_c_ptr(new Graph());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivDeleteGraph(primitivGraph_t *graph) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  delete to_cpp_ptr(graph);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetDefaultGraph(primitivGraph_t **graph) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  *graph = to_c_ptr(&Graph::get_default());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivSetDefaultGraph(primitivGraph_t *graph) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  Graph::set_default(*to_cpp_ptr(graph));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivClearGraph(primitivGraph_t *graph) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  to_cpp_ptr(graph)->clear();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivExecuteGraphForward(
    primitivGraph_t *graph, const primitivNode_t *node,
    const primitivTensor_t **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr(&to_cpp_ptr(graph)->forward(*to_cpp_ptr(node)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivExecuteGraphBackward(
    primitivGraph_t *graph, const primitivNode_t *node) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  to_cpp_ptr(graph)->backward(*to_cpp_ptr(node));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetGraphShape(
    const primitivGraph_t *graph, const primitivNode_t *node,
    primitivShape_t **shape) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  *shape = to_c_ptr_from_value(
      to_cpp_ptr(graph)->get_shape(*to_cpp_ptr(node)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetDeviceFromGraph(
    const primitivGraph_t *graph, const primitivNode_t *node,
    primitivDevice_t **device) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(device);
  *device = to_c_ptr(&to_cpp_ptr(graph)->get_device(*to_cpp_ptr(node)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivDumpGraph(
    const primitivGraph_t *graph, const char *format, char *buffer,
    size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  PRIMITIV_C_CHECK_NOT_NULL(format);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_string_to_array(
      to_cpp_ptr(graph)->dump(format), buffer, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetGraphNumOperators(
    const primitivGraph_t *graph, uint32_t *num) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  PRIMITIV_C_CHECK_NOT_NULL(num);
  *num = to_cpp_ptr(graph)->num_operators();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
