#include <config.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <utility>
#include <primitiv/graph.h>

using std::move;
using std::cout;
using std::endl;

namespace primitiv {

Graph::~Graph() {
  for (ValueNode *v : vals_) {
    delete v;
  }
  for (FunctionNode *f : funcs_) {
    delete f->func;
    delete f;
  }
}

Node Graph::add_function(
    Function *func,
    const std::initializer_list<Node> &args) {
  // Check each argument.
  for (const Node &arg : args) {
    if (arg.g_ != this) {
      std::stringstream ss;
      ss << "Graph mismatched. arg.g_: " << arg.g_ << " != this: " << this;
      throw std::runtime_error(ss.str());
    }
    if (arg.id_ >= vals_.size()) {
      std::stringstream ss;
      ss << "Invalid node ID. "
         << "This may be a bug and the program will abort. "
         << "arg.id_: " << arg.id_
         << " >= vals_.size(): " << vals_.size();
      std::abort();
    }
  }

  // Gather information.
  const unsigned func_id = funcs_.size();
  const unsigned ret_val_id = vals_.size();
  std::vector<unsigned> arg_val_ids;
  std::vector<const Shape *> arg_shapes;
  for (const Node &arg : args) {
    arg_val_ids.emplace_back(arg.id_);
    arg_shapes.emplace_back(&vals_[arg.id_]->shape);
  }
  Shape ret_shape = func->forward_shape(arg_shapes);
  
  // Update graph.
  for (const unsigned arg_val_id : arg_val_ids) {
    vals_[arg_val_id]->sink_func_ids.emplace_back(func_id);
  }
  funcs_.emplace_back(new FunctionNode {func, move(arg_val_ids), ret_val_id});
  vals_.emplace_back(new ValueNode {move(ret_shape), func_id, {}});

  return Node(this, ret_val_id);
}

void Graph::dump() const {
  cout << "Computation graph:" << endl;
  cout << "  Value Nodes:" << endl;
  for (unsigned i = 0; i < vals_.size(); ++i) {
    const ValueNode &v = *vals_[i];
    cout << "    [" << i << "]"
         << ": shape=" << v.shape.to_string()
         << ", src=" << v.src_func_id
         << ", sinks=[";
    for (unsigned j = 0; j < v.sink_func_ids.size(); ++j) {
      if (j > 0) cout << ',';
      cout << v.sink_func_ids[j];
    }
    cout << ']' << endl;
  }
  cout << "  Function Nodes:" << endl;
  for (unsigned i = 0; i < funcs_.size(); ++i) {
    const FunctionNode &f = *funcs_[i];
    cout << "    [" << i << "]"
         << ": func=" << f.func->name()
         << ", args=[";
    for (unsigned j = 0; j < f.arg_val_ids.size(); ++j) {
      if (j > 0) cout << ',';
      cout << f.arg_val_ids[j];
    }
    cout << "], ret=" << f.ret_val_id << endl;
  }
}

}  // namespace primitiv
