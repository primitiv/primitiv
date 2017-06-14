#ifndef PRIMITIV_TRAINER_H_
#define PRIMITIV_TRAINER_H_

namespace primitiv {

class Parameter;

/**
 * Abstract class for parameter optimizers.
 */
class Trainer {
  Trainer(const Trainer &) = delete;
  Trainer(Trainer &&) = delete;
  Trainer &operator=(const Trainer &) = delete;
  Trainer &operator=(Trainer &&) = delete;

public:
  Trainer() = default;
  virtual ~Trainer() = default;

  /**
   * Registers a parameter.
   * @param param Parameter to be optimized.
   */
  virtual void add_parameter(Parameter *param) = 0;

  /**
   * Resets all gradients of registered parameters.
   */
  virtual void reset_gradients() = 0;

  /**
   * Updates parameter values.
   * @param scale Additional learning rate scaling factor.
   */
  virtual void update(float scale) = 0;
};

}  // namespace primitiv

#endif  // PRIMITIV_TRAINER_H_
