#ifndef PRIMITIV_GRAPH_H_
#define PRIMITIV_GRAPH_H_

#include <vector>
#include <primitiv/function.h>
#include <primitiv/node.h>
#include <primitiv/shape.h>

namespace primitiv {

class Device;

/**
 * Computation graph.
 */
class Graph {
  Graph(const Graph &) = delete;
  Graph(Graph &&) = delete;
  Graph &operator=(const Graph &) = delete;
  Graph &operator=(Graph &&) = delete;

public:
  Graph() = default;
  ~Graph();

  /**
   * Adds a function subgraph.
   * @param func Interface of the new function.
   * @param args List of arguments. Each node should point a node in the same
   *        computation graph.
   * @return A new Node object of the resulting value.
   */
  Node add_function(Function *func, const std::vector<Node> &args);

  /**
   * Calculates the value of given node.
   * @param node Node object specifying the target node.
   * @return Calculated value.
   * @remarks This function calculates only the subgraph which is required to
   *          calculate the target node. Each intermediate result is stored to
   *          the corresponding node in the subgraph and they are re-used for
   *          future calculation. I.e., each node is calculated only once while
   *          the lifetime of the Graph object.
   */
  const Tensor &forward(const Node &node);

  /**
   * Calculates the backpropagation.
   * @param node Node object specifying the output node.
   * @remarks `node` should point to a node in the forward path, i.e., the same
   *          node used to call `forward()`, or an ancestor node of that.
   *          Descendant nodes of `node` are removed from the backward path.
   */
  void backward(const Node &node);

  /**
   * Retrieves the shape of the node.
   * @param node Node object specifying the target node.
   * @return The shape of the node.
   */
  const Shape &get_shape(const Node &node) const;

  /**
   * Retrieves the device of the node.
   * @param node Node object specifying the target node.
   * @return the device of the node.
   */
  Device *get_device(const Node &node) const;

  /**
   * Retrieves the value of the node.
   * @param node Node object specifying the target node.
   * @return Calculated value if it is already calculated, or an invalid tensor
   *         otherwise.
   * @remarks This method does not affect the internal information of the graph.
   */
  const Tensor &get_value(const Node &node) const;

  /**
   * Retrieves the gradient of the node.
   * @param node Node object specifying the target node.
   * @return Calculated value if it is already calculated, or an invalid tensor
   *         otherwise.
   * @remarks This method does not affect the internal information of the graph.
   */
  const Tensor &get_gradient(const Node &node) const;

  /**
   * Dump internal graphs.
   */
  void dump() const;

  /**
   * Returns the number of functions in the computation graph.
   * @return Number of nodes.
   */
  unsigned num_functions() const { return funcs_.size(); }

private:
  /**
   * Tuple of values to determine the location of the node.
   */
  struct Address {
    unsigned fid;
    unsigned vid;
  };

  /**
   * Informations of each node.
   */
  struct NodeInfo {
    Shape shape;
    Device *device;
    Tensor *value;
    Tensor *grad;
    std::vector<unsigned> sinks;
  };

  /**
   * Set of informations that represents the function: an implementation of the
   * function, its arguments, and its return values.
   */
  struct FunctionInfo {
    Function *func;
    std::vector<Address> args;
    std::vector<NodeInfo> rets;
  };

  std::vector<FunctionInfo> funcs_;
};

}  // namespace primitiv

#endif  // PRIMITIV_GRAPH_H_
