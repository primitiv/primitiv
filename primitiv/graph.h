#ifndef PRIMITIV_GRAPH_H_
#define PRIMITIV_GRAPH_H_

#include <initializer_list>
#include <vector>
#include <primitiv/function.h>
#include <primitiv/node.h>
#include <primitiv/shape.h>

namespace primitiv {

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
  Node add_function(
      Function *func,
      const std::initializer_list<const Node> &args);

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
   * Returns the number of nodes in the computation graph.
   * @return Number of nodes.
   */
  inline unsigned num_nodes() const { return nodes_.size(); }

private:
  struct NodeInfo {
  private:
    NodeInfo(const NodeInfo &) = delete;
    NodeInfo &operator=(const NodeInfo &) = delete;

  public:
    NodeInfo() = default;
    NodeInfo(NodeInfo &&) = default;
    NodeInfo &operator=(NodeInfo &&) = default;

    Shape shape;
    Function *func;
    Tensor value;
    Tensor grad;
    std::vector<unsigned> args;
    std::vector<unsigned> sinks;
  };

  std::vector<NodeInfo *> nodes_;
};

}  // namespace primitiv

#endif  // PRIMITIV_GRAPH_H_
