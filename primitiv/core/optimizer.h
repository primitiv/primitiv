#ifndef PRIMITIV_CORE_OPTIMIZER_H_
#define PRIMITIV_CORE_OPTIMIZER_H_

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include <primitiv/core/error.h>
#include <primitiv/core/mixins/nonmovable.h>

namespace primitiv {

class Model;
class Parameter;

/**
 * Abstract class for parameter optimizers.
 */
class Optimizer : mixins::Nonmovable<Optimizer> {
public:
  Optimizer() : epoch_(0), lr_scale_(1), l2_strength_(0), clip_threshold_(0) {}

  virtual ~Optimizer() = default;

  /**
   * Loads configurations from a file.
   * @param path Path of the optimizer parameter file.
   */
  void load(const std::string &path);

  /**
   * Saves current configurations to a file.
   * @param path Path of the file that will store optimizer parameters.
   */
  void save(const std::string &path) const;

  /**
   * Retrieves current epoch.
   * @return Current epoch.
   */
  std::uint32_t get_epoch() const { return epoch_; }

  /**
   * Sets current epoch.
   * @param epoch New epoch.
   */
  void set_epoch(std::uint32_t epoch) { epoch_ = epoch; }

  /**
   * Retrieves current learning rate scaling factor.
   * @return The scaling factor.
   */
  float get_learning_rate_scaling() const { return lr_scale_; }

  /**
   * Sets learning rate scaling factor.
   * @param scale New scaling factor.
   * @remarks Could not set negative values.
   */
  void set_learning_rate_scaling(float scale) {
    if (scale < 0) PRIMITIV_THROW_ERROR(
        "Could not set negative value to learning_rate_scaling.");
    lr_scale_ = scale;
  }

  /**
   * Retrieves current L2 decay strength.
   * @return Current L2 decay strength.
   */
  float get_weight_decay() const { return l2_strength_; }

  /**
   * Sets L2 decay strength.
   * @param strength New L2 decay strength, or 0 to disable L2 decay.
   * @remarks Could not set negative values.
   */
  void set_weight_decay(float strength) {
    if (strength < 0) PRIMITIV_THROW_ERROR(
        "Could not set negative value to weight_decay.");
    l2_strength_ = strength;
  }

  /**
   * Retrieves current gradient clipping threshold.
   * @return Current gradient clipping threshold.
   */
  float get_gradient_clipping() const { return clip_threshold_; }

  /**
   * Sets gradient clipping threshold.
   * @param threshold New clipping threshold, or 0 to disable gradient clipping.
   * @remarks Could not set negative values.
   */
  void set_gradient_clipping(float threshold) {
    if (threshold < 0) PRIMITIV_THROW_ERROR(
        "Could not set negative value to gradient_clipping.");
    clip_threshold_ = threshold;
  }

  /**
   * Do nothing.
   * This function is used as the sentinel of other specialized functions.
   */
  void add() { /* do nothing */ }

  /**
   * Registers multiple parameters and models.
   * @param model_or_param Parameter or Model to be optimized.
   * @param args List of remaining Parameter or Model to be optimized.
   *
   * This function behaves similar to multiple `add()` calls with the same
   * order of arguments.
   * E.g., below lines should behave similarly (except the case of exceptions):
   *
   *     add(a, b, c, d);
   *     add(a, b); add(c, d);
   *     add(a); add(b); add(c); add(d);
   */
  template<typename T, typename... Args>
  void add(T &model_or_param, Args &... args) {
    add_inner(model_or_param);
    add(args...);
  }

private:
  /**
   * Registers a parameter (inner function).
   * @param param Parameter to be optimized.
   */
  void add_inner(Parameter &param);

  /**
   * Registers all trainable parameters in a model (inner function).
   * @param model Model to be optimized.
   */
  void add_inner(const Model &model);

public:
  /**
   * Resets all gradients of registered parameters.
   */
  void reset_gradients();

  /**
   * Updates parameter values.
   */
  void update();

  /**
   * Gathers configuration values.
   * @param uint_configs Configurations with std::uint32_t type.
   * @param float_configs Configurations with float type.
   */
  virtual void get_configs(
      std::unordered_map<std::string, std::uint32_t> &uint_configs,
      std::unordered_map<std::string, float> &float_configs) const;

  /**
   * Sets configuration values.
   * @param uint_configs Configurations with std::uint32_t type.
   * @param float_configs Configurations with float type.
   */
  virtual void set_configs(
      const std::unordered_map<std::string, std::uint32_t> &uint_configs,
      const std::unordered_map<std::string, float> &float_configs);

private:
  std::uint32_t epoch_;
  float lr_scale_;
  float l2_strength_;
  float clip_threshold_;

  // TODO(odashi):
  // This lookup table does not work if a different Parameter object is
  // allocated at the same pointer.
  std::unordered_set<Parameter *> params_;

  /**
   * Event handler on adding a new parameter.
   * @param param New Parameter object that is added to the parameter list.
   */
  virtual void configure_parameter(Parameter &param) = 0;

  /**
   * Updates a parameter.
   * @param param Parameter to be updated.
   * @param scale Additional learning rate scaling factor.
   */
  virtual void update_parameter(float scale, Parameter &param) = 0;
};

}  // namespace primitiv

#endif  // PRIMITIV_CORE_OPTIMIZER_H_
