/*
 * NOTE(odashi):
 * Inner structure of Graph is designed to handle multivalued operators for
 * future extensions, but for now this code handels only one results of each
 * operator.
 */

#include <primitiv/config.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <utility>
#include <primitiv/error.h>
#include <primitiv/functions.h>
#include <primitiv/graph.h>

using std::cerr;
using std::cout;
using std::endl;
using std::move;
using std::vector;

namespace primitiv {

void Graph::clear() {
  ops_.clear();
}

#define CHECK_NODE(n) { \
  if ((n).g_ != this) { \
    THROW_ERROR( \
        "Graph mismatched. node.g_: " << (n).g_ << " != this: " << this); \
  } \
  if ((n).op_id_ >= ops_.size() || \
      (n).val_id_ >= ops_[(n).op_id_].rets.size()) { \
    cerr \
        << "Invalid node detected." << endl \
        << "This may be a bug and the program will abort." << endl \
        << "Please report this to the developers. " << endl \
        << "  node.g_: " << (n).g_ << endl \
        << "  node.op_id_: " << (n).op_id_ << endl \
        << "  node.val_id_: " << (n).val_id_ << endl; \
    std::abort(); \
  } \
}

#define ACCESS(n) (ops_[n.op_id_].rets[n.val_id_])

Node Graph::add_operator(
    std::unique_ptr<Operator> &&op, const std::vector<Node> &args) {
  // Gathers information of args.
  vector<Address> arg_addrs(args.size());
  vector<const Shape *> arg_shapes(args.size());
  for (std::uint32_t i = 0; i < args.size(); ++i) {
    const Node &arg = args[i];
    CHECK_NODE(arg);
    arg_addrs[i] = { arg.op_id_, arg.val_id_ };
    arg_shapes[i] = &ACCESS(arg).shape;
  }

  // Calculates the shape of the resulting value.
  // This may throw an exception when trying an invalid operation.
  Shape ret_shape = op->forward_shape(arg_shapes);

  // Retrieves the device object which manages return values itself.
  Device *ret_device = op->get_device();
  if (!ret_device) {
    // If nullptr, the device object is inherited from `args[0]`.
    ret_device = args.size() > 0 ? &ACCESS(args[0]).device : nullptr;
    if (!ret_device) {
      THROW_ERROR(
          "Bad device forwarding of operator '" << op->name()
          << "' with " << args.size() << " argument(s).");
    }
  }

  // Makes nodes of return values.
  vector<NodeInfo> rets;
  rets.emplace_back(NodeInfo {
      move(ret_shape), *ret_device, Tensor(), Tensor(), vector<std::uint32_t>(),
  });

  // Updates the graph.
  const std::uint32_t ret_op_id = ops_.size();
  for (const Address &arg_addr : arg_addrs) {
    ops_[arg_addr.op_id].rets[arg_addr.val_id].sinks.emplace_back(ret_op_id);
  }
  ops_.emplace_back(OperatorInfo { move(op), move(arg_addrs), move(rets) });

  return Node(*this, ret_op_id, 0);
}

const Tensor &Graph::forward(const Node &node) {
  CHECK_NODE(node);

  std::function<const Tensor *(std::uint32_t)> forward_recursive = [&](
      std::uint32_t op_id) -> const Tensor * {
    OperatorInfo &cur_f = ops_[op_id];
    NodeInfo &cur_n = cur_f.rets[0];

    // Try to get the inner value of the operator.
    const Tensor *inner_v = cur_f.op->get_inner_value();
    if (inner_v) return inner_v;

    if (!cur_n.value.valid()) {
      // Gathers arguments.
      vector<const Tensor *> arg_values;
      arg_values.reserve(cur_f.args.size());
      for (const Address &arg : cur_f.args) {
        arg_values.emplace_back(forward_recursive(arg.op_id));
      }

      // Calculates the value.
      cur_n.value = cur_f.op->forward(arg_values);
    }

    return &cur_n.value;
  };

  return *forward_recursive(node.op_id_);
}

void Graph::backward(const Node &node) {
  CHECK_NODE(node);

  OperatorInfo &last_f = ops_[node.op_id_];
  NodeInfo &last_n = last_f.rets[node.val_id_];

  // Check whether the last node is already forwarded or not.
  const Tensor *last_v = last_n.value.valid()
    ? &last_n.value
    : last_f.op->get_inner_value();
  if (!last_v) {
    forward(node);
    last_v = &last_n.value;
    if (!last_v) {
      // NOTE(ocashi): Should never arrive here.
      THROW_ERROR(
          "The node [op_id=" << node.op_id_ << ", val_id=" << node.val_id_
          << "] is not yet forwarded.");
    }
  }

  // Makes the identity gradient (dx/dx = 1) at the last node.
  last_n.grad = functions::ones<Tensor>(last_v->shape(), last_n.device);

  // Performs backpropagation.
  // NOTE(odashi):
  // In the current implementation, the node ID corresponds to the inverse
  // topological order of the computation graph.
  for (std::int32_t op_id = node.op_id_; op_id >= 0; --op_id) {
    OperatorInfo &cur_f = ops_[op_id];
    NodeInfo &cur_n = cur_f.rets[0];
    const Tensor *cur_v = cur_n.value.valid()
      ? &cur_n.value
      : cur_f.op->get_inner_value();

    // If the gradient is invalid, this operator is out of the forward path.
    if (!cur_n.grad.valid()) continue;

    // Gathers argument value/gradient tensors.
    const std::uint32_t arg_size = cur_f.args.size();
    vector<const Tensor *> arg_values;
    vector<Tensor *> arg_grads;
    arg_values.reserve(arg_size);
    arg_grads.reserve(arg_size);
    for (std::uint32_t i = 0; i < arg_size; ++i) {
      const Address &arg = cur_f.args[i];
      OperatorInfo &arg_f = ops_[arg.op_id];
      NodeInfo &arg_n = arg_f.rets[arg.val_id];
      const Tensor *arg_v = arg_n.value.valid()
        ? &arg_n.value
        : arg_f.op->get_inner_value();
      if (!arg_n.grad.valid()) {
        arg_n.grad = functions::zeros<Tensor>(arg_v->shape(), arg_n.device);
      }
      arg_values.emplace_back(arg_v);
      arg_grads.emplace_back(&arg_n.grad);
    }

    // Propagetes the gradient from this node.
    cur_f.op->backward(*cur_v, cur_n.grad, arg_values, arg_grads);

    // Deletes current gradient to suppress memory.
    cur_n.grad.invalidate();
  }
}

Shape Graph::get_shape(const Node &node) const {
  CHECK_NODE(node);
  return ACCESS(node).shape;
}

Device &Graph::get_device(const Node &node) const {
  CHECK_NODE(node);
  return ACCESS(node).device;
}

std::string Graph::dump(const std::string &format) const {
  if (format != "dot") THROW_ERROR("Unknown format: " << format);

  std::stringstream ss;
  ss << R"(digraph ComputationGraph {
  graph [
    fontname = "Times-Roman",
  ];
  node [
    fontname = "Times-Roman",
    shape = "box",
    style = "rounded",
    width = "0.0",
    height = "0.0",
  ];
  edge [
    fontname = "Courier",
  ];)";

  for (std::uint32_t i = 0; i < ops_.size(); ++i) {
    const OperatorInfo &f = ops_[i];
    ss << "  " << i << " [label = \"" << f.op->name() << "\"];\n";
  }

  for (std::uint32_t i = 0; i < ops_.size(); ++i) {
    const OperatorInfo &f = ops_[i];
    for (std::uint32_t j = 0; j < f.args.size(); ++j) {
      const Shape &s = ops_[f.args[j].op_id].rets[f.args[j].val_id].shape;
      ss << "  "
         << f.args[j].op_id << " -> " << i
         << "[label = \"" << s.to_string() << "\"];\n";
    }
  }

  ss << "}\n";
  return ss.str();
}

}  // namespace primitiv
