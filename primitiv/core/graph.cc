#include <primitiv/config.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <utility>

#include <primitiv/core/error.h>
#include <primitiv/core/functions.h>
#include <primitiv/core/graph.h>
#include <primitiv/core/string_utils.h>

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
    PRIMITIV_THROW_ERROR( \
        "Graph mismatched. node.g_: " << (n).g_ << " != this: " << this); \
  } \
  if ((n).oid_ >= ops_.size() || \
      (n).vid_ >= ops_[(n).oid_].rets.size()) { \
    cerr \
        << "Invalid node detected." << endl \
        << "This may be a bug and the program will abort." << endl \
        << "Please report this to the developers. " << endl \
        << "  node.g_: " << (n).g_ << endl \
        << "  node.oid_: " << (n).oid_ << endl \
        << "  node.vid_: " << (n).vid_ << endl; \
    std::abort(); \
  } \
}

vector<Node> Graph::add_operator(
    std::unique_ptr<Operator> &&op, const std::vector<Node> &args) {
  const std::uint32_t argn_req = op->num_arguments();
  const std::uint32_t retn = op->num_returns();
  const std::uint32_t argn = args.size();

  // Checks the number of arguments.
  if (argn_req == Operator::NONZERO) {
    if (argn == 0) {
      PRIMITIV_THROW_ERROR(
          "Invalid number of arguments. required: NONZERO, actual: " << argn);
    }
  } else if (argn_req != Operator::ANY) {
    if (argn != argn_req) {
      PRIMITIV_THROW_ERROR(
          "Invalid number of arguments. required: " << argn_req
          << ", actual: " << argn);
    }
  }

  // Retrieves the device object which manages return values itself.
  Device *ret_device = op->get_device();
  if (!ret_device) {
    // The device object should be inherited from `args[0]`.
    ret_device = argn > 0
      ? ops_[args[0].oid_].rets[args[0].vid_].device
      : nullptr;
    if (!ret_device) {
      PRIMITIV_THROW_ERROR(
          "Bad device propagation occurred. operator: '" << op->name()
          << "', argn: " << argn);
    }
  }

  // Gathers information of arguments.
  vector<Address> arg_addrs(argn);
  vector<const Shape *> arg_shapes(argn);
  for (std::uint32_t i = 0; i < argn; ++i) {
    const Node &arg = args[i];
    CHECK_NODE(arg);
    arg_addrs[i] = { arg.oid_, arg.vid_ };
    arg_shapes[i] = &ops_[arg.oid_].rets[arg.vid_].shape;
  }

  // Makes nodes of return values.
  vector<NodeInfo> rets(retn);
  vector<Shape *> ret_shapes(retn);
  for (std::uint32_t i = 0; i < retn; ++i) {
    rets[i].device = ret_device;
    ret_shapes[i] = &rets[i].shape;
  }

  // Calculates the shape of the resulting value.
  // This may throw an exception when trying an invalid operation.
  op->forward_shape(arg_shapes, ret_shapes);

  // Updates the graph.
  const std::uint32_t ret_oid = ops_.size();
  //for (const Address &arg_addr : arg_addrs) {
  //  ops_[arg_addr.oid].rets[arg_addr.vid].sinks.emplace_back(ret_oid);
  //}
  ops_.emplace_back(OperatorInfo { move(op), move(arg_addrs), move(rets) });

  // Creates Node objects.
  vector<Node> nodes;
  nodes.reserve(retn);
  for (std::uint32_t i = 0; i < retn; ++i) {
    nodes.emplace_back(Node { *this, ret_oid, i });
  }
  return nodes;
}

const Tensor &Graph::forward(const Node &node) {
  CHECK_NODE(node);

  std::function<const Tensor *(const Address)> forward_recursive = [&](
      const Address addr) -> const Tensor * {
    OperatorInfo &cur_f = ops_[addr.oid];

    if (cur_f.op->has_inner_values()) {
      return cur_f.op->get_inner_values()[addr.vid];
    }

    NodeInfo &cur_n = cur_f.rets[addr.vid];
    if (cur_n.value.valid()) {
      return &cur_n.value;
    }

    // Gathers arguments and return values.
    vector<const Tensor *> args_v;
    vector<Tensor *> rets_v;
    args_v.reserve(cur_f.args.size());
    rets_v.reserve(cur_f.rets.size());
    for (const Address arg : cur_f.args) {
      args_v.emplace_back(forward_recursive(arg));
    }
    for (NodeInfo &ret : cur_f.rets) {
      rets_v.emplace_back(&ret.value);
    }

    // Calculates the value.
    cur_f.op->forward(args_v, rets_v);

    return &cur_n.value;
  };

  return *forward_recursive(Address { node.oid_, node.vid_ });
}

void Graph::backward(const Node &node) {
  CHECK_NODE(node);

  OperatorInfo &last_f = ops_[node.oid_];
  NodeInfo &last_n = last_f.rets[node.vid_];

  // Force to perform the forward operation.
  if (!last_n.value.valid()) {
    forward(node);
  }

  // Makes the identity gradient (dx/dx = 1) at the last node.
  last_n.grad = functions::ones<Tensor>(last_n.shape, last_n.device);

  // Performs the backpropagation.
  // NOTE(odashi):
  // In the current implementation, the node ID corresponds to the inverse
  // topological order of the computation graph.
  for (std::int32_t oid = node.oid_; oid >= 0; --oid) {
    OperatorInfo &cur_f = ops_[oid];
    const std::uint32_t argn = cur_f.args.size();
    const std::uint32_t retn = cur_f.rets.size();

    // Gathers information of return values.
    vector<const Tensor *> rets_v(retn), rets_g(retn);
    bool enabled = false;
    for (uint32_t i = 0; i < retn; ++i) {
      NodeInfo &cur_n = cur_f.rets[i];
      rets_v[i] = &cur_n.value;
      rets_g[i] = &cur_n.grad;
      enabled = enabled || cur_n.grad.valid();
    }
    if (!enabled) {
      // This operator is out of the forward path because all gradients of
      // return values are invalid.
      continue;
    }

    // All invalid gradients of return values should be treated as 0.
    for (uint32_t i = 0; i < retn; ++i) {
      NodeInfo &cur_n = cur_f.rets[i];
      if (!cur_n.grad.valid()) {
        cur_n.grad = functions::zeros<Tensor>(cur_n.shape, cur_n.device);
      }
    }

    // Gathers information of arguments.
    vector<const Tensor *> args_v(argn);
    vector<Tensor *> args_g(argn);
    for (uint32_t i = 0; i < argn; ++i) {
      const Address arg = cur_f.args[i];
      OperatorInfo &arg_f = ops_[arg.oid];
      NodeInfo &arg_n = arg_f.rets[arg.vid];
      args_v[i] = arg_n.value.valid()
        ? &arg_n.value
        : arg_f.op->get_inner_values()[arg.vid];
      args_g[i] = &arg_n.grad;
      if (!arg_n.grad.valid()) {
        arg_n.grad = functions::zeros<Tensor>(arg_n.shape, arg_n.device);
      }
    }

    // Propagetes the gradient from this node.
    cur_f.op->backward(args_v, rets_v, rets_g, args_g);

    // Deletes current gradient to suppress memory.
    for (uint32_t i = 0; i < retn; ++i) {
      cur_f.rets[i].grad.invalidate();
    }
  }
}

Shape Graph::get_shape(const Node &node) const {
  CHECK_NODE(node);
  return ops_[node.oid_].rets[node.vid_].shape;
}

Device &Graph::get_device(const Node &node) const {
  CHECK_NODE(node);
  return *ops_[node.oid_].rets[node.vid_].device;
}

std::string Graph::dump(const std::string &format) const {
  if (format != "dot") PRIMITIV_THROW_ERROR("Unknown format: " << format);

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

  // TODO(odashi):
  // Following implementation does not distinguish multiple return values.

  for (std::uint32_t i = 0; i < ops_.size(); ++i) {
    const OperatorInfo &f = ops_[i];
    ss << "  " << i << " [label = \"" << f.op->name() << "\"];\n";
  }

  for (std::uint32_t i = 0; i < ops_.size(); ++i) {
    const OperatorInfo &f = ops_[i];
    for (std::uint32_t j = 0; j < f.args.size(); ++j) {
      const Shape &s = ops_[f.args[j].oid].rets[f.args[j].vid].shape;
      ss << "  "
         << f.args[j].oid << " -> " << i
         << "[label = \"" << s.to_string() << "\"];\n";
    }
  }

  ss << "}\n";
  return ss.str();
}

}  // namespace primitiv
