/*
 * NOTE(odashi):
 * Inner structure of Graph is designed to handle multivalued functions for
 * future extensions, but for now this code handels only one results of each
 * function.
 */

#include <config.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <utility>
#include <primitiv/device.h>
#include <primitiv/error.h>
#include <primitiv/function.h>
#include <primitiv/graph.h>

using std::cerr;
using std::cout;
using std::endl;
using std::move;
using std::vector;

namespace primitiv {

Graph *Graph::default_graph_ = nullptr;

Graph &Graph::get_default_graph() {
  if (!default_graph_) THROW_ERROR("Default graph is null.");
  return *default_graph_;
}

void Graph::set_default_graph(Graph &g) {
  default_graph_ = &g;
}

Graph::~Graph() {
  if (default_graph_ == this) {
    default_graph_ = nullptr;
  }
}

#define CHECK_NODE(n) { \
  if ((n).g_ != this) { \
    THROW_ERROR( \
        "Graph mismatched. node.g_: " << (n).g_ << " != this: " << this); \
  } \
  if ((n).fid_ >= funcs_.size() || \
      (n).vid_ >= funcs_[(n).fid_].rets.size()) { \
    cerr \
        << "Invalid node detected." << endl \
        << "This may be a bug and the program will abort." << endl \
        << "Please report this to the developers. " << endl \
        << "  node.g_: " << (n).g_ << endl \
        << "  node.fid_: " << (n).fid_ << endl \
        << "  node.vid_: " << (n).vid_ << endl; \
    std::abort(); \
  } \
}

#define ACCESS(n) (funcs_[n.fid_].rets[n.vid_])

Node Graph::add_function(
    std::unique_ptr<Function> &&func, const std::vector<Node> &args) {
  // Gathers information of args.
  vector<Address> arg_addrs(args.size());
  vector<const Shape *> arg_shapes(args.size());
  for (unsigned i = 0; i < args.size(); ++i) {
    const Node &arg = args[i];
    CHECK_NODE(arg);
    arg_addrs[i] = { arg.fid_, arg.vid_ };
    arg_shapes[i] = &ACCESS(arg).shape;
  }

  // Calculates the shape of the resulting value.
  // This may throw an exception when trying an invalid operation.
  Shape ret_shape = func->forward_shape(arg_shapes);

  // Retrieves the device object which manages return values itself.
  Device *ret_device = func->get_device();
  if (!ret_device) {
    // If nullptr, the device object is inherited from `args[0]`.
    ret_device = args.size() > 0 ? &ACCESS(args[0]).device : nullptr;
    if (!ret_device) {
      THROW_ERROR(
          "Bad device forwarding of function '" << func->name()
          << "' with " << args.size() << " argument(s).");
    }
  }

  // Make nodes of return values.
  vector<NodeInfo> rets;
  rets.emplace_back(NodeInfo {
      move(ret_shape), *ret_device,
      std::unique_ptr<Tensor>(), std::unique_ptr<Tensor>(),
      vector<unsigned>() });

  // Updates the graph.
  const unsigned ret_fid = funcs_.size();
  for (const Address &arg_addr : arg_addrs) {
    funcs_[arg_addr.fid].rets[arg_addr.vid].sinks.emplace_back(ret_fid);
  }
  funcs_.emplace_back(FunctionInfo { move(func), move(arg_addrs), move(rets) });

  return Node(*this, ret_fid, 0);
}

const Tensor &Graph::forward(const Node &node) {
  CHECK_NODE(node);

  std::function<const Tensor *(unsigned)> forward_recursive = [&](
      unsigned fid) -> const Tensor * {
    FunctionInfo &f = funcs_[fid];

    // Try to get the inner value of the function.
    const Tensor *v = f.func->get_inner_value();

    // Check whether the function is already calculated or not.
    // NOTE(odashi):
    // Once the function is traversed, the gradient tensor becomes not nullptr.
    if (f.rets[0].grad) return v ? v : f.rets[0].value.get();

    if (!v) {
      // The function does not have own values,
      // but can calculate it via forward().

      // Gathers arguments.
      vector<const Tensor *> arg_values(f.args.size());
      for (unsigned i = 0; i < f.args.size(); ++i) {
        const Address &arg = f.args[i];
        arg_values[i] = forward_recursive(arg.fid);
      }

      // Calculates the value.
      f.rets[0].value.reset(new Tensor(f.func->forward(arg_values)));
      v = f.rets[0].value.get();
    }

    // Resets gradients.
    f.rets[0].grad.reset(new Tensor(v->device().new_tensor(v->shape(), 0)));
    return v;
  };

  return *forward_recursive(node.fid_);
}

void Graph::backward(const Node &node) {
  CHECK_NODE(node);

  FunctionInfo &last_f = funcs_[node.fid_];
  if (!last_f.rets[0].grad) {
    THROW_ERROR("Node is still not calculated in the forward path.");
  }

  // Make the identity gradient (dx/dx = 1) at the last node.
  last_f.rets[node.vid_].grad->reset(1);

  // NOTE(odashi):
  // In the current implementation, the node ID corresponds to the inverse
  // topological order of the computation graph.

  // Performs backpropagation.
  for (int fid = node.fid_; fid >= 0; --fid) {
    const FunctionInfo &cur_f = funcs_[fid];

    // If the gradient is nullptr, the function is out of the forward path.
    if (!cur_f.rets[0].grad) continue;

    // Gather argument value/gradient tensors.
    const unsigned arg_size = cur_f.args.size();
    vector<const Tensor *> arg_values(arg_size);
    vector<Tensor *> arg_grads(arg_size);
    for (unsigned i = 0; i < arg_size; ++i) {
      const Address &arg = cur_f.args[i];
      NodeInfo &arg_n = funcs_[arg.fid].rets[arg.vid];
      const Tensor *v = funcs_[arg.fid].func->get_inner_value();
      arg_values[i] = v ? v : arg_n.value.get();
      arg_grads[i] = arg_n.grad.get();
    }

    // Propagetes the gradient from this node.
    const NodeInfo &cur_n = cur_f.rets[0];
    cur_f.func->backward(*cur_n.value, *cur_n.grad, arg_values, arg_grads);
  }
}

const Shape &Graph::get_shape(const Node &node) const {
  CHECK_NODE(node);
  return ACCESS(node).shape;
}

Device &Graph::get_device(const Node &node) const {
  CHECK_NODE(node);
  return ACCESS(node).device;
}

const Tensor &Graph::get_value(const Node &node) const {
  CHECK_NODE(node);
  const FunctionInfo &f = funcs_[node.fid_];
  const NodeInfo &r = f.rets[node.vid_];

  // Check gradient existence to check whether the value is calculated or not.
  if (!r.grad) THROW_ERROR("Node is still not calculated.");

  const Tensor *v = f.func->get_inner_value();
  return v ? *v : *r.value;
}

const Tensor &Graph::get_gradient(const Node &node) const {
  CHECK_NODE(node);
  const std::unique_ptr<Tensor> &ret = ACCESS(node).grad;
  if (!ret) THROW_ERROR("Node is still not calculated.");
  return *ret;
}

void Graph::dump() const {
  cout << "Computation graph:" << endl;
  for (unsigned i = 0; i < funcs_.size(); ++i) {
    const FunctionInfo &f = funcs_[i];
    cout << "Function " << i
         << ": name=" << f.func->name()
         << ", args=[";
    for (unsigned j = 0; j < f.args.size(); ++j) {
      if (j > 0) cout << ", ";
      cout << f.args[j].fid << ':' << f.args[j].vid;
    }
    cout << ']' << endl;
    for (unsigned j = 0; j < f.rets.size(); ++j) {
      const NodeInfo &n = f.rets[j];
      cout << "  Return " << j
           << ": shape=" << n.shape.to_string()
           << ", sinks=[";
      for (unsigned k = 0; k < n.sinks.size(); ++k) {
        if (k > 0) cout << ", ";
        cout << n.sinks[k];
      }
      cout << ']' << endl;
    }
  }
}

}  // namespace primitiv
