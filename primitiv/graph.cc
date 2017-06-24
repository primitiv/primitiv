#include <config.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <utility>
#include <primitiv/device.h>
#include <primitiv/error.h>
#include <primitiv/graph.h>

using std::cerr;
using std::cout;
using std::endl;
using std::move;
using std::vector;

namespace primitiv {

Graph::~Graph() {
  // Removes all allocated objects.
  for (FunctionInfo &f : funcs_) {
    delete f.func;
    for (NodeInfo &n : f.rets) {
      delete n.value;
      delete n.grad;
    }
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

Node Graph::add_function(Function *func, const std::vector<Node> &args) {
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
  // TODO(odashi): fix this
  vector<Shape> ret_shapes { func->forward_shape(arg_shapes) };

  // Retrieves the device object which manages return values itself.
  Device *ret_device = func->get_device();
  if (!ret_device) {
    // If nullptr, the device object is inherited from `args[0]`.
    ret_device = args.size() > 0 ? ACCESS(args[0]).device : nullptr;
    if (!ret_device) {
      THROW_ERROR(
          "Bad device forwarding of function '" << func->name()
          << "' with " << args.size() << " argument(s).");
    }
  }

  // Make nodes of return values.
  const unsigned NUM_NODES = 1;  // TODO(odashi): fix this
  vector<NodeInfo> rets(
      NUM_NODES,
      NodeInfo { Shape(), ret_device, nullptr, nullptr, vector<unsigned>() });
  for (unsigned i = 0; i < NUM_NODES; ++i) {
    rets[i].shape = move(ret_shapes[i]);
  }

  // Updates the graph.
  const unsigned ret_fid = funcs_.size();
  for (const Address &arg_addr : arg_addrs) {
    funcs_[arg_addr.fid].rets[arg_addr.vid].sinks.emplace_back(ret_fid);
  }
  funcs_.emplace_back(FunctionInfo { func, move(arg_addrs), move(rets) });

  return Node(this, ret_fid, 0);
}

const Tensor &Graph::forward(const Node &node) {
  CHECK_NODE(node);

  std::function<void(unsigned)> forward_recursive = [&](unsigned fid) {
    FunctionInfo &f = funcs_[fid];

    // Check whether the function is already calculated or not.
    if (f.rets[0].value) return;

    // Gathers arguments.
    vector<const Tensor *> arg_values(f.args.size());
    for (unsigned i = 0; i < f.args.size(); ++i) {
      const Address &arg = f.args[i];
      forward_recursive(arg.fid);
      arg_values[i] = funcs_[arg.fid].rets[arg.vid].value;
    }

    // Calculates results.
    // TODO(odashi): fix this.
    vector<Tensor *> ret_values { new Tensor(f.func->forward(arg_values)) };
    const unsigned NUM_NODES = 1;
    for (unsigned i = 0; i < NUM_NODES; ++i) {
      f.rets[i].value = ret_values[i];
      f.rets[i].grad = new Tensor(
          ret_values[i]->device()->new_tensor(ret_values[i]->shape(), 0));
    }
  };

  forward_recursive(node.fid_);
  return *ACCESS(node).value;
}

void Graph::backward(const Node &node) {
  CHECK_NODE(node);

  FunctionInfo &last_f = funcs_[node.fid_];
  if (!last_f.rets[0].value) {
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

    // If the value is nullptr, the function is out of the forward path.
    if (!cur_f.rets[0].value) continue;

    // Gather argument value/gradient tensors.
    const unsigned arg_size = cur_f.args.size();
    vector<const Tensor *> arg_values(arg_size);
    vector<Tensor *> arg_grads(arg_size);
    for (unsigned i = 0; i < arg_size; ++i) {
      const Address &arg = cur_f.args[i];
      NodeInfo &arg_n = funcs_[arg.fid].rets[arg.vid];
      arg_values[i] = arg_n.value;
      arg_grads[i] = arg_n.grad;
    }

    // Propagetes the gradient from this node.
    // TODO(odashi): fix this.
    const NodeInfo &cur_n = cur_f.rets[0];
    cur_f.func->backward(*cur_n.value, *cur_n.grad, arg_values, arg_grads);
  }
}

const Shape &Graph::get_shape(const Node &node) const {
  CHECK_NODE(node);
  return ACCESS(node).shape;
}

Device *Graph::get_device(const Node &node) const {
  CHECK_NODE(node);
  return ACCESS(node).device;
}

const Tensor &Graph::get_value(const Node &node) const {
  CHECK_NODE(node);
  const Tensor *ret = ACCESS(node).value;
  if (!ret) THROW_ERROR("Node is still not calculated.");
  return *ret;
}

const Tensor &Graph::get_gradient(const Node &node) const {
  CHECK_NODE(node);
  const Tensor *ret = ACCESS(node).grad;
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
