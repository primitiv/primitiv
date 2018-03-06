#ifndef PRIMITIV_PLUGIN_FUNCTION_H_
#define PRIMITIV_PLUGIN_FUNCTION_H_

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

#include <primitiv/device.h>
#include <primitiv/dynamic_library.h>
#include <primitiv/graph.h>
#include <primitiv/mixins.h>
#include <primitiv/operator.h>
#include <primitiv/string_utils.h>

namespace primitiv {

/**
 * Interface of precompiled functions.
 */
class PluginFunction : mixins::Noncopyable<PluginFunction> {
public:
  /**
   * Initializes a new function using a specified binary.
   * @param path Path to the dynamic library file containing the implementation
   *             of the precompiled function.
   * @param dev Device
   * @throw primitiv::Error Initialization not succeeded.
   */
  explicit PluginFunction(const std::string &path)
  : lib_(path)
  , num_args_fp_(lib_.get_symbol<std::uint32_t(void)>("num_arguments"))
  , num_rets_fp_(lib_.get_symbol<std::uint32_t(void)>("num_returns"))
  , fwd_shp_fp_(lib_.get_symbol<
      void(const Shape *const *const, Shape *const *const)>("forward_shape"))
  , fwd_fp_(lib_.get_symbol<
      void(const Tensor *const *const, Tensor *const *const)>("forward"))
  , bwd_fp_(lib_.get_symbol<
      void(
        const Tensor *const *const, const Tensor *const *const,
        const Tensor *const *const, Tensor *const *const)>("backward")) {}

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

  /**
   * Directly calls the forward operation using argument lists.
   * @param head The first argument.
   * @param tail List of remaining arguments.
   * @return List of return values.
   */
  template<typename... Args>
  std::vector<Tensor> operator()(
      const Tensor &head, const Args &...tail) const {
    const std::uint32_t argn = num_arguments();
    const std::uint32_t retn = num_returns();
    if (sizeof...(tail) + 1 != argn) {
      PRIMITIV_THROW_ERROR(
          "Invalid number of arguments. expected: "
          << argn << ", actual: " << (sizeof...(tail) + 1));
    }

    const std::vector<const Tensor *> args_vp { &head, &tail... };

    std::vector<Shape> args_s(argn), rets_s(retn);
    std::vector<const Shape *> args_sp(argn);
    std::vector<Shape *> rets_sp(retn);
    for (std::uint32_t i = 0; i < argn; ++i) {
      args_s[i] = args_vp[i]->shape();
      args_sp[i] = &args_s[i];
    }
    for (std::uint32_t i = 0; i < retn; ++i) {
      rets_sp[i] = &rets_s[i];
    }

    fwd_shp_fp_(args_sp.data(), rets_sp.data());

    std::vector<Tensor> rets_v(retn);
    std::vector<Tensor *> rets_vp(retn);
    for (std::uint32_t i = 0; i < retn; ++i) {
      rets_vp[i] = &rets_v[i];
    }

    fwd_fp_(args_vp.data(), rets_vp.data());
    return rets_v;
  }

  class Operator : public primitiv::Operator {
  public:
    explicit Operator(const PluginFunction &pf) : pf_(pf) {}
    std::string name() const override {
      return
        "Plugin("
        + string_utils::to_string(
            reinterpret_cast<std::uintptr_t>(pf_.lib_.handle()))
        + ')';
    }

    std::uint32_t num_arguments() const override {
      return pf_.num_args_fp_();
    }

    std::uint32_t num_returns() const override {
      return pf_.num_rets_fp_();
    }

    bool has_inner_values() const override { return false; }

    void forward_shape(
        const std::vector<const Shape *> &args,
        const std::vector<Shape *> &rets) const override {
      pf_.fwd_shp_fp_(args.data(), rets.data());
    }

    void forward(
        const std::vector<const Tensor *> &args,
        const std::vector<Tensor *> &rets) const override {
      pf_.fwd_fp_(args.data(), rets.data());
    }

    void backward(
        const std::vector<const Tensor *> &args_v,
        const std::vector<const Tensor *> &rets_v,
        const std::vector<const Tensor *> &rets_g,
        const std::vector<Tensor *> &args_g) const override {
      pf_.bwd_fp_(args_v.data(), rets_v.data(), rets_g.data(), args_g.data());
    }

  private:
    const PluginFunction &pf_;
  };

  /**
   * Registers a new operator representing the plugin function to the graph.
   * @param head The first argument.
   * @param tail List of remaining arguments.
   * @return List of return values.
   */
  template<typename... Args>
  std::vector<Node> operator()(
      const Node &head, const Args &...tail) const {
    const std::uint32_t argn = num_arguments();
    if (sizeof...(tail) + 1 != argn) {
      PRIMITIV_THROW_ERROR(
          "Invalid number of arguments. expected: "
          << argn << ", actual: " << (sizeof...(tail) + 1));
    }
    return head.graph().add_operator(
        std::unique_ptr<Operator>(new PluginFunction::Operator(*this)),
        { head, tail... });
  }

private:
  DynamicLibrary lib_;
  std::function<std::uint32_t(void)> num_args_fp_;
  std::function<std::uint32_t(void)> num_rets_fp_;
  std::function<
    void(const Shape *const *const, Shape *const *const)> fwd_shp_fp_;
  std::function<
    void(const Tensor *const *const, Tensor *const *const)> fwd_fp_;
  std::function<
    void(
        const Tensor *const *const, const Tensor *const *const,
        const Tensor *const *const, Tensor *const *const)> bwd_fp_;
};

}  // namespace primitiv

#endif  // PRIMITIV_PLUGIN_FUNCTION_H_
