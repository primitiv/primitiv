#ifndef PRIMITIV_NODE_H_
#define PRIMITIV_NODE_H_

#include <primitiv/error.h>

namespace primitiv {

class Graph;

/**
 * Pointer of a node in the computation graph.
 */
class Node {
  friend Graph;

public:
  inline Node() : g_(), func_id_(), val_id_() {}
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
   * Returns the function ID.
   * @return Function ID.
   */
  inline unsigned function_id() const {
    if (!g_) THROW_ERROR("Invalid node.");
    return func_id_;
  }

  /**
   * Returns the value ID of the function.
   * @return Value ID.
   */
  inline unsigned value_id() const {
    if(!g_) THROW_ERROR("Invalid node.");
    return val_id_;
  }

private:
  /**
   * Creates a new node pointer.
   * @param g Pointer of the computation graph.
   * @param function_id Function ID.
   * @param value_id Value ID.
   */
  inline Node(Graph *g, unsigned function_id, unsigned value_id)
  : g_(g), func_id_(function_id), val_id_(value_id) {}

  Graph *g_;
  unsigned func_id_;
  unsigned val_id_;
};

}  // namespace primitiv

#endif  // PRIMITIV_NODE_H_
