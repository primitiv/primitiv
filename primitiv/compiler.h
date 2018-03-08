#ifndef PRIMITIV_COMPILER_H_
#define PRIMITIV_COMPILER_H_

#include <cstddef>
#include <string>
#include <vector>

#include <primitiv/error.h>
#include <primitiv/plugin_function.h>

namespace primitiv {

class Device;

class Compiler {
public:
  class Node {
    friend class Compiler;
  public:
    /**
     * Creats an invalid Node object.
     */
    Node() : cp_(nullptr), fid_(0), vid_(0) {}

    /**
     * Retrieves whether the Node object is valid or not.
     * @return true if the Node is valid, false otherwise.
     */
    bool valid() const { return cp_ != nullptr; }

    /**
     * Checks whether the Node object is valid or not.
     * @throw primitiv::Error The Node is invalid.
     */
    void check_valid() const {
      if (!valid()) {
        PRIMITIV_THROW_ERROR("Invalid Compiler::Node.");
      }
    }

    /**
     * Retrieves the Compiler object corresponding to the Node object.
     * @return Reference of the Compiler object.
     */
    Compiler &compiler() const {
      check_valid();
      return *cp_;
    }

  private:
    Compiler *cp_;
    std::size_t fid_;
    std::size_t vid_;

    /**
     * Creates a Node object using specific members.
     * @param cp Compiler object corresponding to the node.
     * @param fid Function ID.
     * @param vid Value ID.
     */
    Node(Compiler &cp, std::size_t fid, std::size_t vid)
    : cp_(&cp), fid_(fid), vid_(vid) {}
  };

  /**
   * Creates a new Compiler object.
   * @param num_arguments Number of arguments of the new function.
   * @param num_returns Number of return values of the new function.
   */
  Compiler(std::size_t num_arguments, std::size_t num_results);

  /**
   * Returns a Node object representing the i-th argument.
   * @param index Index of the argument.
   * @return Node object representing the i-th argument.
   * @throw primitiv::Error Invalid index is given.
   */
  Compiler::Node input(std::size_t index);

  /**
   * Registers a Node object as the i-th return value.
   * @param index Index of the return value.
   * @param node Node object to be registered.
   * @throw primitiv::Error Invalid index is given,
   *                        or invalid node is given,
   *                        or the output is already set.
   */
  void output(std::size_t index, const Compiler::Node &node);

  /**
   * Adds a new operator into the compiler.
   * @param name Name of the operator.
   * @params args List of Argument nodes.
   * @reutrn List of return values.
   */
  std::vector<Compiler::Node> add_operator(
      const std::string &name, const std::vector<Compiler::Node> &args);

  /**
   * Compiles the graph and generates a new PluginFunction object.
   * @param device Target device.
   * @param output_directory Path to the directory to put generated files.
   * @return New PluginFunction object.
   */
  PluginFunction compile(
      Device &device, const std::string &output_dierectory) const;

private:
  struct Address {
    std::size_t fid;
    std::size_t vid;
  };

  struct OperatorInfo {
    std::string name;
    std::vector<Address> args;
    std::size_t retn;
  };

  struct OutputInfo {
    bool valid;
    Address addr;
  };

  std::vector<OperatorInfo> ops_;
  std::vector<OutputInfo> outputs_;
};

Compiler::Node operator+(const Compiler::Node &a, const Compiler::Node &b) {
  return a.compiler().add_operator("add", {a, b})[0];
}

}  // namespace primitiv

#endif  // PRIMITIV_COMPILER_H_
