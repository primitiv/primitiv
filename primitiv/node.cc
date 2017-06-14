#include <config.h>

#include <primitiv/graph.h>
#include <primitiv/node.h>

namespace primitiv {

const Shape &Node::shape() const {
  if (!g_) THROW_ERROR("Invalid node.");
  return g_->get_shape(*this);
}

const Tensor &Node::value() const {
  if (!g_) THROW_ERROR("Invalid node.");
  return g_->get_value(*this);
}

const Tensor &Node::gradient() const {
  if (!g_) THROW_ERROR("Invalid node.");
  return g_->get_gradient(*this);
}

}  // namespace primitiv
