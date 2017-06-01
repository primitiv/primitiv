#include <config.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <utility>
#include <primitiv/device.h>
#include <primitiv/error.h>
#include <primitiv/graph.h>

using std::move;
using std::cout;
using std::endl;
using std::vector;

namespace primitiv {

Graph::~Graph() {
  for (NodeInfo *n : nodes_) {
    delete n->func;
    delete n;
  }
}

#define CHECK_NODE(node) { \
  if ((node).g_ != this) { \
    THROW_ERROR( \
        "Graph mismatched. node.g_: " << (node).g_ << " != this: " << this); \
  } \
  if ((node).id_ >= nodes_.size()) { \
    THROW_ERROR( \
        "Invalid node ID. " \
        << "This may be a bug and the program will abort. " \
        << "node.id_: " << (node).id_ \
        << " >= nodes_.size(): " << nodes_.size()); \
  } \
}

Node Graph::add_function(
    Function *func,
    const std::initializer_list<const Node> &args) {
  // Gathers information of args.
  vector<unsigned> arg_ids;
  vector<const Shape *> arg_shapes;
  for (const Node &arg : args) {
    CHECK_NODE(arg);
    arg_ids.emplace_back(arg.id_);
    arg_shapes.emplace_back(&nodes_[arg.id_]->shape);
  }

  // Calculates the shape of the resulting value.
  // This may throw an exception when trying an invalid operation.
  Shape ret_shape = func->forward_shape(arg_shapes);

  // Updates the graph.
  const unsigned ret_id = nodes_.size();
  for (const unsigned arg_id : arg_ids) {
    nodes_[arg_id]->sinks.emplace_back(ret_id);
  }
  NodeInfo *node = new NodeInfo;
  node->shape = move(ret_shape);
  node->func = func;
  node->args = move(arg_ids);
  nodes_.emplace_back(node);

  return Node(this, ret_id);
}

const Tensor &Graph::forward(const Node &node) {
  CHECK_NODE(node);

  std::function<void(const unsigned)> forward_recursive = [&](
      const unsigned id) {
    NodeInfo &n = *nodes_[id];
    if (n.value.valid()) {
      // Values of the node are already calculated.
      return;
    }

    // First time of accessing this node. Calculates actual values.
    vector<const Tensor *> args;
    for (const unsigned arg_id : n.args) {
      forward_recursive(arg_id);
      args.emplace_back(&nodes_[arg_id]->value);
    }
    n.value = n.func->forward(args);
  };

  forward_recursive(node.id_);
  return nodes_[node.id_]->value;
}

void Graph::backward(const Node &node) {
  CHECK_NODE(node);

  NodeInfo &last_node = *nodes_[node.id_];
  if (!last_node.value.valid()) {
    THROW_ERROR(
        "Node " << node.id_ << " is not calculated in the forward path.");
  }
  if (last_node.grad.valid()) {
    THROW_ERROR("Node " << node.id_ << " already has the gradient vector.");
  }

  // Make identity gradient at the last node.
  last_node.grad = last_node.value.device()->new_tensor(last_node.shape);
  last_node.grad.reset(1);

  // The node ID represents the topological order.
  for (int id = node.id_; id >= 0; --id) {
    const NodeInfo &cur_node = *nodes_[id];
    if (!cur_node.value.valid()) {
      // Not calculated in the forward path.
      continue;
    }

    // Gather argument value/gradient tensors.
    vector<const Tensor *> arg_values;
    vector<Tensor *> arg_grads;
    for (unsigned arg : cur_node.args) {
      NodeInfo &arg_node = *nodes_[arg];
      if (!arg_node.grad.valid()) {
        arg_node.grad = arg_node.value.device()->new_tensor(arg_node.shape);
        arg_node.grad.reset(0);
      }
      arg_values.emplace_back(&arg_node.value);
      arg_grads.emplace_back(&arg_node.grad);
    }

    // Propagetes the gradient from this node.
    cur_node.func->backward(
        cur_node.value, cur_node.grad, arg_values, arg_grads);
  }
}

const Tensor &Graph::get_value(const Node &node) const {
  CHECK_NODE(node);
  return nodes_[node.id_]->value;
}

const Tensor &Graph::get_gradient(const Node &node) const {
  CHECK_NODE(node);
  return nodes_[node.id_]->grad;
}

void Graph::dump() const {
  cout << "Computation graph:" << endl;
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    const NodeInfo &n = *nodes_[i];
    cout << "  [" << i << "]"
         << ": shape=" << n.shape.to_string()
         << ", func=" << n.func->name()
         << ", args=[";
    for (unsigned j = 0; j < n.args.size(); ++j) {
      if (j > 0) cout << ',';
      cout << n.args[j];
    }
    cout << "], sinks=[";
    for (unsigned j = 0; j < n.sinks.size(); ++j) {
      if (j > 0) cout << ',';
      cout << n.sinks[j];
    }
    cout << ']' << endl;
  }
}

}  // namespace primitiv
