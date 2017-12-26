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

/**
 * Opaque type of Node.
 */
typedef struct primitiv_Node primitiv_Node;

/**
 * Opaque type of Graph.
 */
typedef struct primitiv_Graph primitiv_Graph;

/**
 * Creates a new Node object.
 * @param node Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Node_new(primitiv_Node **node);

/**
 * Creates a clone of existing Node object.
 * @param src Pointer to a source Node.
 * @param node Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Node_clone(
    primitiv_Node *src, primitiv_Node **node);

/**
 * Deletes the Node object.
 * @param node Pointer of a handler.
 */
extern PRIMITIV_C_API void primitiv_Node_delete(primitiv_Node *node);

/**
 * Returns whether the node is valid or not.
 * @param node Pointer of a handler.
 * @return true or false w.r.t. the node is valid or not.
 */
extern PRIMITIV_C_API bool primitiv_Node_valid(const primitiv_Node *node);

/**
 * Returns corresponding Graph object.
 * @param node Pointer of a handler.
 * @param graph Pointer to receive the Graph object.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Node_graph(
    const primitiv_Node *node, primitiv_Graph **graph);

/**
 * Returns the operator ID.
 * @param node Pointer of a handler.
 * @param id Pointer to receive the operator ID.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Node_operator_id(
    const primitiv_Node *node, uint32_t *id);

/**
 * Returns the value ID of the operator.
 * @param node Pointer of a handler.
 * @param id Pointer to receive the value ID.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Node_value_id(
    const primitiv_Node *node, uint32_t *id);

/**
 * Returns shape of the node.
 * @param node Pointer of a handler.
 * @param shape Pointer to receive a Shape object.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Node_shape(
    const primitiv_Node *node, primitiv_Shape **shape);

/**
 * Returns device of the node.
 * @param node Pointer of a handler.
 * @param device Pointer to receive the Device object.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Node_device(
    const primitiv_Node *node, primitiv_Device **device);

/**
 * Calculates the value of this node and returns a float.
 * @param node Pointer of a handler.
 * @param value Pointer to receive a calculated float value.
 * @return Status code.
 * @remarks This function calls Graph::forward() internally.
 *          This function can be used only when the Node has a scalar and
 *          non-minibatched shape (i.e., shape() == Shape())
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Node_to_float(
    const primitiv_Node *node, float *value);

/**
 * Calculates the value of this node and returns a list of float.
 * @param node Pointer of a handler.
 * @param array Pointer to receive a list of calculated values.
 * @return Status code.
 * @remarks This function calls Graph::forward() internally.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Node_to_array(
    const primitiv_Node *node, float *array);

/**
 * Returns argmax indices along an axis of this node.
 * @param node Pointer of a handler.
 * @param dim A specified axis.
 * @param indices Pointer to receive a list of integers that indicates positions
 *                of the maximum values.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Node_argmax(
    const primitiv_Node *node, uint32_t dim, uint32_t *indices);

/**
 * Returns argmin indices along an axis of this node.
 * @param node Pointer of a handler.
 * @param dim A specified axis.
 * @param indices Pointer to receive a list of integers that indicates positions
 *                of the minimum values.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Node_argmin(
    const primitiv_Node *node, uint32_t dim, uint32_t *indices);

/**
 * Executes the backward operation from this node.
 * @param node Pointer of a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Node_backward(
    const primitiv_Node *node);

/**
 * Creates a new Graph object.
 * @param graph Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Graph_new(
    primitiv_Graph **graph);

/**
 * Deletes the Graph object.
 * @param graph Pointer of a handler.
 */
extern PRIMITIV_C_API void primitiv_Graph_delete(primitiv_Graph *graph);

/**
 * Retrieves the current default graph.
 * @param graph Pointer to receive the current default graph.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Graph_get_default(
    primitiv_Graph **graph);

/**
 * Specifies a new default graph.
 * @param graph Pointer of the new default graph.
 */
extern PRIMITIV_C_API void primitiv_Graph_set_default(primitiv_Graph *graph);

/**
 * Clear all operators in the graph.
 * @param graph Pointer of a handler.
 * @remarks After calling this method, all Node objects supplied by the graph
 *          itself is invalidated.
 */
extern PRIMITIV_C_API void primitiv_Graph_clear(primitiv_Graph *graph);

/**
 * Calculates the value of given node.
 * @param graph Pointer of a handler.
 * @param node Node object specifying the target node.
 * @param tensor Pointer to receive a calculated value.
 * @return Status code.
 * @remarks This function calculates only the subgraph which is required to
 *          calculate the target node. Each intermediate result is stored to
 *          the corresponding node in the subgraph and they are re-used for
 *          future calculation. I.e., each node is calculated only once while
 *          the lifetime of the Graph object.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Graph_forward(
    primitiv_Graph *graph, const primitiv_Node *node,
    const primitiv_Tensor **tensor);

/**
 * Calculates the backpropagation.
 * @param graph Pointer of a handler.
 * @param node Node object specifying the output node.
 * @return Status code.
 * @remarks If `node` is not yet forwarded, this function implicitly calls
 *          `forward(node)`.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Graph_backward(
    primitiv_Graph *graph, const primitiv_Node *node);

/**
 * Retrieves the shape of the node.
 * @param graph Pointer of a handler.
 * @param node Node object specifying the target node.
 * @param shape Pointer to receive the shape of the node.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Graph_get_shape(
    const primitiv_Graph *graph, const primitiv_Node *node,
    primitiv_Shape **shape);

/**
 * Retrieves the device of the node.
 * @param graph Pointer of a handler.
 * @param node Node object specifying the target node.
 * @param device Pointer to receive the device of the node.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Graph_get_device(
    const primitiv_Graph *graph, const primitiv_Node *node,
    primitiv_Device **device);

/**
 * Dump internal graph structure.
 * @param graph Pointer of a handler.
 * @param format Name of the format. Available options:
 *                 "dot" ... Graphviz's dot format.
 * @param string Pointer to receive a string that represents the internal graph
 *               using given format.
 * @param length Pointer to receive a length of the string.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Graph_dump(
    const primitiv_Graph *graph, const char *format, char *string,
    size_t *length);

/**
 * Returns the number of operators in the computation graph.
 * @param graph Pointer of a handler.
 * @return Number of nodes.
 */
extern PRIMITIV_C_API uint32_t primitiv_Graph_num_operators(
    const primitiv_Graph *graph);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_GRAPH_H_
