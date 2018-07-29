#include <primitiv/config.h>

#include <string>
#include <vector>

#include <primitiv/core/graph.h>
#include <primitiv/c/internal/internal.h>
#include <primitiv/c/graph.h>

using primitiv::Node;
using primitiv::Graph;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;
using primitiv::c::internal::to_c_ptr_from_value;

PRIMITIV_C_STATUS primitivCreateNode(primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Node);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCloneNode(
    const primitivNode_t *src, primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(src);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Node(*to_cpp_ptr(src)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivDeleteNode(primitivNode_t *node) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  delete to_cpp_ptr(node);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivIsValidNode(
    const primitivNode_t *node, PRIMITIV_C_BOOL *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(node)->valid();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetGraphFromNode(
    const primitivNode_t *node, primitivGraph_t **retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_c_ptr(&to_cpp_ptr(node)->graph());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetNodeOperatorId(
    const primitivNode_t *node, uint32_t *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(node)->operator_id();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetNodeValueId(
    const primitivNode_t *node, uint32_t *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(node)->value_id();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetNodeShape(
    const primitivNode_t *node, primitivShape_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(to_cpp_ptr(node)->shape());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetDeviceFromNode(
    const primitivNode_t *node, primitivDevice_t **retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_c_ptr(&to_cpp_ptr(node)->device());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivEvaluateNodeAsFloat(
    const primitivNode_t *node, float *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(node)->to_float();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivEvaluateNodeAsArray(
    const primitivNode_t *node, float *retval, size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_vector_to_array(
      to_cpp_ptr(node)->to_vector(), retval, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetNodeArgmax(
    const primitivNode_t *node, uint32_t dim, uint32_t *retval,
    size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_vector_to_array(
      to_cpp_ptr(node)->argmax(dim), retval, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetNodeArgmin(
    const primitivNode_t *node, uint32_t dim, uint32_t *retval,
    size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_vector_to_array(
      to_cpp_ptr(node)->argmin(dim), retval, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivExecuteNodeBackward(const primitivNode_t *node) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  to_cpp_ptr(node)->backward();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateGraph(primitivGraph_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Graph());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivDeleteGraph(primitivGraph_t *graph) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  delete to_cpp_ptr(graph);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetDefaultGraph(primitivGraph_t **retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_c_ptr(&Graph::get_default());
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
    const primitivTensor_t **retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_c_ptr(&to_cpp_ptr(graph)->forward(*to_cpp_ptr(node)));
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
    primitivShape_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      to_cpp_ptr(graph)->get_shape(*to_cpp_ptr(node)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetDeviceFromGraph(
    const primitivGraph_t *graph, const primitivNode_t *node,
    primitivDevice_t **retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_c_ptr(&to_cpp_ptr(graph)->get_device(*to_cpp_ptr(node)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivDumpGraph(
    const primitivGraph_t *graph, const char *format, char *retval,
    size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  PRIMITIV_C_CHECK_NOT_NULL(format);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_string_to_array(
      to_cpp_ptr(graph)->dump(format), retval, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetGraphNumOperators(
    const primitivGraph_t *graph, uint32_t *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(graph);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(graph)->num_operators();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
