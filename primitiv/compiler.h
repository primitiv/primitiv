#ifndef PRIMITIV_COMPILER_H_
#define PRIMITIV_COMPILER_H_

#include <cstdint>

#include <primitiv/device.h>
#include <primitiv/plugin_function.h>

namespace primitiv {

class Compiler {
public:
  class Node {
  private:
    Compiler *cp_;
    std::uint32_t fid_;
    std::uint32_t vid_;
  };

  /**
   * Creates a new Compiler object.
   * @param num_arguments Number of arguments of the new function.
   * @param num_returns Number of return values of the new function.
   */
  Compiler(std::uint32_t num_arguments, std::uint32_t num_results) = default;

  /**
   * Returns a Node object representing the i-th argument.
   * @param index Index of the argument.
   * @return Node object representing the i-th argument.
   */
  Compiler::Node input(std::uint32_t index);

  /**
   * Registers a Node object as the i-th return value.
   * @param index Index of the return value.
   * @param node Node object to be registered.
   */
  void output(std::uint32_t index, const Compiler::Node &node);

  /**
   * Compiles the graph and generates a new PluginFunction object.
   * @return New PluginFunction object.
   */
  PluginFunction compile() const;

private:

};

}  // namespace primitiv

#endif  // PRIMITIV_COMPILER_H_
