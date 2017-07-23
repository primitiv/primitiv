#ifndef PRIMITIV_TRAINER_H_
#define PRIMITIV_TRAINER_H_

#include <unordered_map>
#include <primitiv/error.h>

namespace primitiv {

class Parameter;

/**
 * Abstract class for parameter optimizers.
 */
class Trainer {
public:
  Trainer(const Trainer &) = default;
  Trainer(Trainer &&) = default;
  Trainer &operator=(const Trainer &) = default;
  Trainer &operator=(Trainer &&) = default;
  virtual ~Trainer() = default;

  Trainer() : epoch_(0), lr_scale_(1), l2_strength_(0), clip_threshold_(0) {}

  /**
   * Retrieves current epoch.
   * @return Current epoch.
   */
  unsigned get_epoch() const { return epoch_; }

  /**
   * Sets current epoch.
   * @param epoch New epoch.
   */
  void set_epoch(unsigned epoch) { epoch_ = epoch; }

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
    if (scale < 0) THROW_ERROR(
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
    if (strength < 0) THROW_ERROR(
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
    if (threshold < 0) THROW_ERROR(
        "Could not set negative value to gradient_clipping.");
    clip_threshold_ = threshold;
  }

  /**
   * Registers a parameter.
   * @param param Parameter to be optimized.
   */
  void add_parameter(Parameter &param);

  /**
   * Resets all gradients of registered parameters.
   */
  void reset_gradients();

  /**
   * Updates parameter values.
   */
  void update();

private:
  unsigned epoch_;
  float lr_scale_;
  float l2_strength_;
  float clip_threshold_;
  std::unordered_map<std::string, Parameter *> params_;

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

#endif  // PRIMITIV_TRAINER_H_
