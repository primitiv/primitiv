#include <primitiv/config.h>

#include <iostream>
#include <unordered_map>

#include <primitiv/compiler.h>
#include <primitiv/device.h>
#include <primitiv/error.h>

namespace {

struct Snippet {
  std::size_t argn;
  std::size_t retn;
  std::string fwd_op;
  std::string bwd_op;
};

std::unordered_map<std::string, Snippet> snippets {
  {"add",
    {
      2, 1,
      "y[0] = x[0] + x[1];",
      "gx[0] += gy[0]; gx[1] += gy[0];",
    }},
  {"input",
    {
      0, 1,
      "",
      "",
  }},
};

#define CHECK_NODE(node) { \
  if (node.cp_ != this \
      || node.fid_ >= ops_.size() \
      || node.vid_ >= ops_[node.fid_].retn) { \
    PRIMITIV_THROW_ERROR( \
        "Invalid node. cp_: " << node.cp_ << ", fid_: " << node.fid_ \
        << ", vid_: " << node.vid_); \
  } \
}

}  // namespace

namespace primitiv {

Compiler::Compiler(std::size_t num_arguments, std::size_t num_returns)
: ops_(num_arguments, OperatorInfo {"input", {}, 1})
, outputs_(num_returns, OutputInfo {false, {}}) {}

Compiler::Node Compiler::input(std::size_t index) {
  if (index >= ops_.size() || ops_[index].name != "input") {
    PRIMITIV_THROW_ERROR("Invalid input index: " << index);
  }
  return Node(*this, index, 0);
}

void Compiler::output(std::size_t index, const Compiler::Node &node) {
  CHECK_NODE(node);
  if (index >= outputs_.size()) {
    PRIMITIV_THROW_ERROR("Invalid output index: " << index);
  }
  if (outputs_[index].valid) {
    PRIMITIV_THROW_ERROR("Output is already set. index: " << index);
  }
  outputs_[index] = OutputInfo {true, {node.fid_, node.vid_}};
}

std::vector<Compiler::Node> Compiler::add_operator(
    const std::string &name, const std::vector<Compiler::Node> &args) {
  const auto snippet_it = ::snippets.find(name);
  if (snippet_it == ::snippets.end()) {
    PRIMITIV_THROW_ERROR("Invalid operator name: " << name);
  }
  const Snippet &snippet = snippet_it->second;
  if (args.size() != snippet.argn) {
    PRIMITIV_THROW_ERROR(
        "Invalid number of arguments. expected: "
        << snippet.argn << ", actual: " << args.size());
  }
  for (const Compiler::Node &arg : args) {
    CHECK_NODE(arg);
  }
  const std::size_t fid = ops_.size();
  ops_.emplace_back(OperatorInfo {name, {}, snippet.retn});
  for (const Compiler::Node &arg : args) {
    ops_.back().args.emplace_back(Address {arg.fid_, arg.vid_});
  }
  std::vector<Compiler::Node> rets;
  rets.reserve(snippet.retn);
  for (std::size_t i = 0; i < snippet.retn; ++i) {
    rets.emplace_back(Node(*this, fid, i));
  }
  return rets;
}

PluginFunction Compiler::compile(
    Device &device, const std::string &output_directory) const {
  using std::cerr;
  using std::endl;

  cerr << "outputs:";
  for (const OutputInfo &oi : outputs_) {
    if (!oi.valid) {
      PRIMITIV_THROW_ERROR("Not all outputs are valid.");
    }
    cerr << " " << oi.addr.fid << ":" << oi.addr.vid;
  }
  cerr << endl;

  for (std::size_t i = 0; i < ops_.size(); ++i) {
    const OperatorInfo &op = ops_[i];
    cerr << "[" << i << "] ";
    cerr << "(" << op.name << ")";
    for (const Address &arg : op.args) {
      cerr << " " << arg.fid << ":" << arg.vid;
    }
    cerr << " ->";
    for (std::size_t j = 0; j < op.retn; ++j) {
      cerr << " " << i << ":" << j;
    }
    cerr << endl;
  }
  PRIMITIV_THROW_ERROR(
      "Test: devptr=" << &device << ", odir=" << output_directory);
}

}  // namespace primitiv
