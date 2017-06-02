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
  inline Node() : g_(), fid_(), vid_() {}
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
    return fid_;
  }

  /**
   * Returns the value ID of the function.
   * @return Value ID.
   */
  inline unsigned value_id() const {
    if(!g_) THROW_ERROR("Invalid node.");
    return vid_;
  }

private:
  /**
   * Creates a new node pointer.
   * @param g Pointer of the computation graph.
   * @param fid Function ID.
   * @param vid Value ID.
   */
  inline Node(Graph *g, unsigned fid, unsigned vid)
    : g_(g), fid_(fid), vid_(vid) {}

  Graph *g_;
  unsigned fid_;
  unsigned vid_;
};

}  // namespace primitiv

#endif  // PRIMITIV_NODE_H_
