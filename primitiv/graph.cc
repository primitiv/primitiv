#include <config.h>

#include <cstdlib>
#include <stdexcept>
#include <sstream>
#include <utility>
#include <primitiv/graph.h>

using std::move;

namespace primitiv {

Node Graph::add_function(
    std::unique_ptr<Function> &&func,
    const std::vector<Node> &args) {
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

  /* check function here */

  // Update graph.
  const unsigned func_id = funcs_.size();
  const unsigned ret_val_id = vals_.size();
  std::vector<unsigned> arg_val_ids;
  for (const Node &arg : args) {
    vals_[arg.id_].sink_func_ids.emplace_back(func_id);
    arg_val_ids.emplace_back(arg.id_);
  }
  funcs_.emplace_back(FunctionNode {move(func), move(arg_val_ids), ret_val_id});
  vals_.emplace_back(ValueNode {func_id, {}});

  return Node(this, ret_val_id);
}

}  // namespace primitiv
