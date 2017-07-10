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
  Trainer() = default;
  Trainer(const Trainer &) = default;
  Trainer(Trainer &&) = default;
  Trainer &operator=(const Trainer &) = default;
  Trainer &operator=(Trainer &&) = default;
  virtual ~Trainer() = default;

  /**
   * Registers a parameter.
   * @param param Parameter to be optimized.
   */
  void add_parameter(Parameter *param);

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

  /**
   * Updates internal states of the trainer.
   */
  virtual void update_epoch() = 0;
};

}  // namespace primitiv

#endif  // PRIMITIV_TRAINER_H_
