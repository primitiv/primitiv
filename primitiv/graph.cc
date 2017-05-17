#include <config.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <utility>
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
    std::stringstream ss; \
    ss << "Graph mismatched. node.g_: " << (node).g_ << " != this: " << this; \
    throw std::runtime_error(ss.str()); \
  } \
  if ((node).id_ >= nodes_.size()) { \
    std::stringstream ss; \
    ss << "Invalid node ID. " \
       << "This may be a bug and the program will abort. " \
       << "node.id_: " << (node).id_ \
       << " >= nodes_.size(): " << nodes_.size(); \
    std::abort(); \
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
  nodes_.emplace_back(
      new NodeInfo {move(ret_shape), func, Tensor(), move(arg_ids), {}});

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
#error
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
