#ifndef PRIMITIV_C_GRAPH_H_
#define PRIMITIV_C_GRAPH_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>
#include <primitiv/c/shape.h>
#include <primitiv/c/tensor.h>

/**
 * Opaque type of Node.
 */
typedef struct primitivNode primitivNode_t;

/**
 * Opaque type of Graph.
 */
typedef struct primitivGraph primitivGraph_t;

/**
 * Creates a new Node object.
 * @param node Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateNode(primitivNode_t **node);

/**
 * Creates a clone of existing Node object.
 * @param src Pointer to a source Node.
 * @param node Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCloneNode(
    primitivNode_t *src, primitivNode_t **node);

/**
 * Deletes the Node object.
 * @param node Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivDeleteNode(primitivNode_t *node);

/**
 * Returns whether the node is valid or not.
 * @param node Pointer of a handler.
 * @param valid Pointer to receive a result: true or false w.r.t. the node is
 *              valid or not.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivIsValidNode(
    const primitivNode_t *node, PRIMITIV_C_BOOL *valid);

/**
 * Returns corresponding Graph object.
 * @param node Pointer of a handler.
 * @param graph Pointer to receive the Graph object.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetGraphFromNode(
    const primitivNode_t *node, primitivGraph_t **graph);

/**
 * Returns the operator ID.
 * @param node Pointer of a handler.
 * @param id Pointer to receive the operator ID.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetNodeOperatorId(
    const primitivNode_t *node, uint32_t *id);

/**
 * Returns the value ID of the operator.
 * @param node Pointer of a handler.
 * @param id Pointer to receive the value ID.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetNodeValueId(
    const primitivNode_t *node, uint32_t *id);

/**
 * Returns shape of the node.
 * @param node Pointer of a handler.
 * @param shape Pointer to receive a Shape object.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetNodeShape(
    const primitivNode_t *node, primitivShape_t **shape);

/**
 * Returns device of the node.
 * @param node Pointer of a handler.
 * @param device Pointer to receive the Device object.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetDeviceFromNode(
    const primitivNode_t *node, primitivDevice_t **device);

/**
 * Calculates the value of this node and returns a float.
 * @param node Pointer of a handler.
 * @param value Pointer to receive a calculated float value.
 * @return Status code.
 * @remarks This function calls Graph::forward() internally.
 *          This function can be used only when the Node has a scalar and
 *          non-minibatched shape (i.e., shape() == Shape())
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivEvaluateNodeAsFloat(
    const primitivNode_t *node, float *value);

/**
 * Calculates the value of this node and returns a list of float.
 * @param node Pointer of a handler.
 * @param array Pointer to receive a list of calculated values.
 * @param size Pointer to receive the length of the array.
 * @return Status code.
 * @remarks This function calls Graph::forward() internally.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivEvaluateNodeAsArray(
    const primitivNode_t *node, float *array, size_t *size);

/**
 * Returns argmax indices along an axis of this node.
 * @param node Pointer of a handler.
 * @param dim A specified axis.
 * @param indices Pointer to receive a list of integers that indicates positions
 *                of the maximum values.
 * @param size Pointer to receive the number of the received indices.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetNodeArgmax(
    const primitivNode_t *node, uint32_t dim, uint32_t *indices,
    size_t *size);

/**
 * Returns argmin indices along an axis of this node.
 * @param node Pointer of a handler.
 * @param dim A specified axis.
 * @param indices Pointer to receive a list of integers that indicates positions
 *                of the minimum values.
 * @param size Pointer to receive the number of the received indices.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetNodeArgmin(
    const primitivNode_t *node, uint32_t dim, uint32_t *indices,
    size_t *size);

/**
 * Executes the backward operation from this node.
 * @param node Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivExecuteNodeBackward(
    const primitivNode_t *node);

/**
 * Creates a new Graph object.
 * @param graph Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateGraph(
    primitivGraph_t **graph);

/**
 * Deletes the Graph object.
 * @param graph Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivDeleteGraph(
    primitivGraph_t *graph);

/**
 * Retrieves the current default graph.
 * @param graph Pointer to receive the current default graph.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetDefaultGraph(
    primitivGraph_t **graph);

/**
 * Specifies a new default graph.
 * @param graph Pointer of the new default graph.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivSetDefaultGraph(
    primitivGraph_t *graph);

/**
 * Clear all operators in the graph.
 * @param graph Pointer of a handler.
 * @remarks After calling this method, all Node objects supplied by the graph
 *          itself is invalidated.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivClearGraph(
    primitivGraph_t *graph);

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
PRIMITIV_C_API PRIMITIV_C_STATUS primitivExecuteGraphForward(
    primitivGraph_t *graph, const primitivNode_t *node,
    const primitivTensor_t **tensor);

/**
 * Calculates the backpropagation.
 * @param graph Pointer of a handler.
 * @param node Node object specifying the output node.
 * @return Status code.
 * @remarks If `node` is not yet forwarded, this function implicitly calls
 *          `forward(node)`.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivExecuteGraphBackward(
    primitivGraph_t *graph, const primitivNode_t *node);

/**
 * Retrieves the shape of the node.
 * @param graph Pointer of a handler.
 * @param node Node object specifying the target node.
 * @param shape Pointer to receive the shape of the node.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetGraphShape(
    const primitivGraph_t *graph, const primitivNode_t *node,
    primitivShape_t **shape);

/**
 * Retrieves the device of the node.
 * @param graph Pointer of a handler.
 * @param node Node object specifying the target node.
 * @param device Pointer to receive the device of the node.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetDeviceFromGraph(
    const primitivGraph_t *graph, const primitivNode_t *node,
    primitivDevice_t **device);

/**
 * Dump internal graph structure.
 * @param graph Pointer of a handler.
 * @param format Name of the format. Available options:
 *                 "dot" ... Graphviz's dot format.
 * @param buffer Pointer to receive a string that represents the internal graph
 *               using given format.
 * @param size Pointer to receive a length of the string.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivDumpGraph(
    const primitivGraph_t *graph, const char *format, char *buffer,
    size_t *size);

/**
 * Returns the number of operators in the computation graph.
 * @param graph Pointer of a handler.
 * @param num Pointer to receive the number of nodes.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetGraphNumOperators(
    const primitivGraph_t *graph, uint32_t *num);

#endif  // PRIMITIV_C_GRAPH_H_
