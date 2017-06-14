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
  Node() : g_(), fid_(), vid_() {}

  /**
   * Returns corresponding Graph object.
   * @return Graph object.
   */
  Graph *graph() const { return g_; }

  /**
   * Returns the function ID.
   * @return Function ID.
   */
  unsigned function_id() const {
    if (!g_) THROW_ERROR("Invalid node.");
    return fid_;
  }

  /**
   * Returns the value ID of the function.
   * @return Value ID.
   */
  unsigned value_id() const {
    if(!g_) THROW_ERROR("Invalid node.");
    return vid_;
  }

  /**
   * Returns shape of the node.
   * @return A Shape object.
   */
  const Shape &shape() const;

  /**
   * Returns the value of the node.
   * @return A Tensor object if the node has been forwarded.
   */
  const Tensor &value() const;

  /**
   * Returns the gradient of the node.
   * @return A Tensor object if the node has been backwarded.
   */
  const Tensor &gradient() const;

private:
  /**
   * Creates a new node pointer.
   * @param g Pointer of the computation graph.
   * @param fid Function ID.
   * @param vid Value ID.
   */
  Node(Graph *g, unsigned fid, unsigned vid) : g_(g), fid_(fid), vid_(vid) {}

  Graph *g_;
  unsigned fid_;
  unsigned vid_;
};

}  // namespace primitiv

#endif  // PRIMITIV_NODE_H_
