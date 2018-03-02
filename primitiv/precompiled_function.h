#ifndef PRIMITIV_PRECOMPILED_FUNCTION_H_
#define PRIMITIV_PRECOMPILED_FUNCTION_H_

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

#include <primitiv/device.h>
#include <primitiv/dynamic_library.h>
#include <primitiv/mixins.h>

namespace primitiv {

/**
 * Interface of precompiled functions.
 */
class PrecompiledFunction : mixins::Noncopyable<PrecompiledFunction> {
public:
  /**
   * Initializes a new function using a specified binary.
   * @param path Path to the dynamic library file containing the implementation
   *             of the precompiled function.
   * @param dev Device
   * @throw primitiv::Error Initialization not succeeded.
   */
  explicit PrecompiledFunction(const std::string &path);

  /**
   * Retrieves the number of required arguments.
   * @return Number of required arguments, or following special values:
   *           Operator::ANY: Arbitrary number of arguments, including 0
   *           Operator::NONZERO: Arbitrary number of arguments, except 0
   */
  std::uint32_t num_arguments() const {
    return num_args_fp_();
  }

  /**
   * Retrieves the number of return values.
   * @return Number of return values.
   */
  std::uint32_t num_returns() const {
    return num_rets_fp_();
  }

  /**
   * Performs the forward operation about shapes.
   * @param args List of argument shapes.
   * @param rets List of resulting shapes.
   * @remarks ``args`` and ``rets`` should have the same number of pointers with
   *          the values returned by ``num_arguments()`` and ``num_returns()``.
   */
  void forward_shape(
      const std::vector<const Shape *> &args,
      const std::vector<Shape *> &rets) const {
    fwd_shp_fp_(args.data(), rets.data());
  }

  /**
   * Performs the forward operation.
   * @param args List of argument values.
   * @param rets List of resulting values.
   * @remarks ``args`` and ``rets`` should have the same number of pointers with
   *          the values returned by ``num_arguments()`` and ``num_returns()``.
   */
  void forward(
      const std::vector<const Tensor *> &args,
      const std::vector<Tensor *> &rets) const {
    fwd_fp_(args.data(), rets.data());
  }

  /**
   * Performs the backward operation.
   * @param args_v List of argument values.
   * @param rets_v List of resulting values.
   * @param rets_g List of gradients of results.
   * @param args_v List of gradients of arguments.
   * @remarks ``args_v/g`` and ``rets_v/g`` should have the same number of
   *          pointers with the values returned by ``num_arguments()`` and
   *          ``num_returns()``.
   */
  void backward(
      const std::vector<const Tensor *> &args_v,
      const std::vector<const Tensor *> &rets_v,
      const std::vector<const Tensor *> &rets_g,
      const std::vector<Tensor *> &args_g) const {
    bwd_fp_(args_v.data(), rets_v.data(), rets_g.data(), args_g.data());
  }

private:
  DynamicLibrary lib_;
  std::function<std::uint32_t(void)> num_args_fp_;
  std::function<std::uint32_t(void)> num_rets_fp_;
  std::function<
    void(const Shape * const * const, Shape * const * const)> fwd_shp_fp_;
  std::function<
    void(const Tensor * const * const, Tensor * const * const)> fwd_fp_;
  std::function<
    void(
        const Tensor * const * const, const Tensor * const * const,
        const Tensor * const * const, Tensor * const * const)> bwd_fp_;
};

}  // namespace primitiv

#endif  // PRIMITIV_PRECOMPILED_FUNCTION_H_
