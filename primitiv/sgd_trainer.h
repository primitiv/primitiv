#ifndef PRIMITIV_SGD_TRAINER_H_
#define PRIMITIV_SGD_TRAINER_H_

#include <vector>
#include <primitiv/trainer.h>

namespace primitiv {

/**
 * Simple stochastic gradient descent.
 */
class SGDTrainer : public Trainer {
  SGDTrainer() = delete;
  SGDTrainer(const SGDTrainer &) = delete;
  SGDTrainer(SGDTrainer &&) = delete;
  SGDTrainer &operator=(const SGDTrainer &) = delete;
  SGDTrainer &operator=(SGDTrainer &&) = delete;

public:
  /**
   * Creates a new SGDTrainer object.
   * @param eta Learning rate.
   */
  explicit SGDTrainer(const float eta) : eta_(eta) {}
  ~SGDTrainer() = default;

  void add_parameter(Parameter *param) override;
  void reset_gradients() override;
  void update() override;

private:
  float eta_;
  std::vector<Parameter *> params_;
};

}  // namespace primitiv

#endif  // PRIMITIV_SGD_TRAINER_H_
