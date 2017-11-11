#ifndef PRIMITIV_GRAPH_H_
#define PRIMITIV_GRAPH_H_

#include <cstdint>
#include <memory>
#include <vector>
#include <primitiv/function.h>
#include <primitiv/mixins.h>
#include <primitiv/shape.h>

namespace primitiv {

class Device;
class Graph;
class Node;

/**
 * Pointer of a node in the computation graph.
 */
class Node {
  friend Graph;

public:
  Node(const Node &) = default;

  Node(Node &&src) : g_(src.g_), fid_(src.fid_), vid_(src.vid_) {
    src.g_ = nullptr;
  }

  Node &operator=(const Node &) = default;

  Node &operator=(Node &&src) {
    if (&src != this) {
      g_ = src.g_;
      fid_ = src.fid_;
      vid_ = src.vid_;
      src.g_ = nullptr;
    }
    return *this;
  }

  Node() : g_(nullptr), fid_(), vid_() {}

  /**
   * Returns whether the node is valid or not.
   * @return true or false w.r.t. the node is valid or not.
   */
  bool valid() const { return !!g_; }

  /**
   * Returns corresponding Graph object.
   * @return Graph object.
   */
  Graph &graph() const {
    if (!valid()) THROW_ERROR("Invalid node.");
    return *g_;
  }

  /**
   * Returns the function ID.
   * @return Function ID.
   */
  std::uint32_t function_id() const {
    if (!valid()) THROW_ERROR("Invalid node.");
    return fid_;
  }

  /**
   * Returns the value ID of the function.
   * @return Value ID.
   */
  std::uint32_t value_id() const {
    if (!valid()) THROW_ERROR("Invalid node.");
    return vid_;
  }

  /**
   * Returns shape of the node.
   * @return A Shape object.
   */
  const Shape &shape() const;

  /**
   * Returns device of the node.
   * @return Device object.
   */
  Device &device() const;

  /**
   * Calculates the value of this node and returns a float.
   * @return A calculated float value.
   * @remarks This function calls Graph::forward() internally.
   *          This function can be used only when the Node has a scalar and
   *          non-minibatched shape (i.e., shape() == Shape())
   */
  float to_float() const;

  /**
   * Calculates the value of this node and returns a list of float.
   * @return A list of calculated values.
   * @remarks This function calls Graph::forward() internally.
   */
  std::vector<float> to_vector() const;

  /**
   * Returns argmax indices along an axis of this node.
   * @param dim A specified axis.
   * @return A list of integers that indicates positions of the maximum values.
   */
  std::vector<std::uint32_t> argmax(std::uint32_t dim) const;

  /**
   * Returns argmin indices along an axis of this node.
   * @param dim A specified axis.
   * @return A list of integers that indicates positions of the minimum values.
   */
  std::vector<std::uint32_t> argmin(std::uint32_t dim) const;

  /**
   * Executes the backward operation from this node.
   */
  void backward() const;

private:
  /**
   * Creates a new node pointer.
   * @param g Pointer of the computation graph.
   * @param fid Function ID.
   * @param vid Value ID.
   */
  Node(Graph &g, std::uint32_t fid, std::uint32_t vid) : g_(&g), fid_(fid), vid_(vid) {}

  Graph *g_;
  std::uint32_t fid_;
  std::uint32_t vid_;
};

/**
 * Computation graph.
 */
class Graph
    : public mixins::DefaultSettable<Graph>
    , mixins::Nonmovable<Graph> {
public:
  Graph() = default;
  ~Graph() = default;

  /**
   * Clear all functions in the graph.
   * @remarks After calling this method, all Node objects supplied by the graph
   *          itself is invalidated.
   */
  void clear();

  /**
   * Adds a function subgraph.
   * @param func Interface of the new function.
   * @param args List of arguments. Each node should point a node in the same
   *        computation graph.
   * @return A new Node object of the resulting value.
   */
  Node add_function(
      std::unique_ptr<Function> &&func, const std::vector<Node> &args);

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
   * @remarks If `node` is not yet forwarded, this function implicitly calls
   *          `forward(node)`.
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
   * @return Device of the node.
   */
  Device &get_device(const Node &node) const;

  /**
   * Dump internal graph structure.
   * @param format Name of the format. Available options:
   *                 "dot" ... Graphviz's dot format.
   * @return A string that represents the internal graph using given format.
   */
  std::string dump(const std::string &format) const;

  /**
   * Returns the number of functions in the computation graph.
   * @return Number of nodes.
   */
  std::uint32_t num_functions() const { return funcs_.size(); }

private:
  /**
   * Tuple of values to determine the location of the node.
   */
  struct Address {
    std::uint32_t fid;
    std::uint32_t vid;
  };

  /**
   * Informations of each node.
   */
  struct NodeInfo {
    Shape shape;
    Device &device;
    Tensor value;
    Tensor grad;
    std::vector<std::uint32_t> sinks;
  };

  /**
   * Set of informations that represents the function: an implementation of the
   * function, its arguments, and its return values.
   */
  struct FunctionInfo {
    std::unique_ptr<Function> func;
    std::vector<Address> args;
    std::vector<NodeInfo> rets;
  };

  static Graph *default_obj_;
  std::vector<FunctionInfo> funcs_;
};

inline const Shape &Node::shape() const {
  if (!valid()) THROW_ERROR("Invalid node.");
  return g_->get_shape(*this);
}

inline Device &Node::device() const {
  if (!valid()) THROW_ERROR("Invalid node.");
  return g_->get_device(*this);
}

inline float Node::to_float() const {
  if (!valid()) THROW_ERROR("Invalid node.");
  return g_->forward(*this).to_float();
}

inline std::vector<float> Node::to_vector() const {
  if (!valid()) THROW_ERROR("Invalid node.");
  return g_->forward(*this).to_vector();
}

inline std::vector<std::uint32_t> Node::argmax(std::uint32_t dim) const {
  if (!valid()) THROW_ERROR("Invalid node.");
  return g_->forward(*this).argmax(dim);
}

inline std::vector<std::uint32_t> Node::argmin(std::uint32_t dim) const {
  if (!valid()) THROW_ERROR("Invalid node.");
  return g_->forward(*this).argmin(dim);
}

inline void Node::backward() const {
  if (!valid()) THROW_ERROR("Invalid node.");
  g_->backward(*this);
}

}  // namespace primitiv

#endif  // PRIMITIV_GRAPH_H_
