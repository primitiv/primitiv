#ifndef PRIMITIV_NODE_H_
#define PRIMITIV_NODE_H_

#include <stdexcept>

namespace primitiv {

class Graph;

/**
 * Pointer of a node in the computation graph.
 */
class Node {
  friend Graph;

public:
  inline Node() : g_(), id_() {}
  Node(const Node &) = default;
  Node(Node &&) = default;
  Node &operator=(const Node &) = default;
  Node &operator=(Node &&) = default;
  ~Node() = default;

  /**
   * Returns corresponding Graph object.
   * @return Graph object.
   */
  inline Graph *graph() const { return g_; }

  /**
   * Returns the node ID.
   * @return Node ID.
   */
  inline unsigned id() const {
    if (!g_) throw std::runtime_error("Invalid node.");
    return id_;
  }

private:
  /**
   * Creates a new node pointer.
   * @param g Pointer of the computation graph.
   * @param id Node ID.
   */
  inline Node(Graph *g, const unsigned id) : g_(g), id_(id) {}

  Graph *g_;
  unsigned id_;
};

}  // namespace primitiv

#endif  // PRIMITIV_NODE_H_
