#ifndef PRIMITIV_TRAINER_H_
#define PRIMITIV_TRAINER_H_

#include <unordered_map>

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

  Trainer() : epoch_(0), l2_strength_(0), clip_threshold_(0) {}

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
   * Retrieves current L2 decay strength.
   * @return Current L2 decay strength.
   */
  float get_weight_decay() const { return l2_strength_; }

  /**
   * Sets L2 decay strength.
   * @param strength New L2 decay strength.
   * @remarks L2 decay will be enabled only if the strength is greater than 0.
   */
  void set_weight_decay(float strength) { l2_strength_ = strength; }

  /**
   * Retrieves current gradient clipping threshold.
   * @return Current gradient clipping threshold.
   */
  float get_gradient_clipping() const { return clip_threshold_; }

  /**
   * Sets gradient clipping threshold.
   * @param threshold New clipping threshold.
   * @remarks Gradient clipping will be enabled only if the threshold is greater
   *          than 0.
   */
  void set_gradient_clipping(float threshold) { clip_threshold_ = threshold; }

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
   * @param scale Additional learning rate scaling factor.
   */
  void update(float scale);

private:
  unsigned epoch_;
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
