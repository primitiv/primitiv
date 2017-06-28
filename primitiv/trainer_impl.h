#ifndef PRIMITIV_TRAINER_IMPL_H_
#define PRIMITIV_TRAINER_IMPL_H_

#include <primitiv/trainer.h>

namespace primitiv {
namespace trainers {

/**
 * Simple stochastic gradient descent.
 */
class SGD : public primitiv::Trainer {
  SGD() = delete;
  SGD(const SGD &) = delete;
  SGD(SGD &&) = delete;
  SGD &operator=(const SGD &) = delete;
  SGD &operator=(SGD &&) = delete;

public:
  /**
   * Creates a new SGD object.
   * @param eta Learning rate.
   */
  explicit SGD(const float eta) : eta_(eta) {}

  void reset_gradients() override;
  void update(float scale) override;

private:
  float eta_;
};

}  // namespace trainers
}  // namespace primitiv

#endif  // PRIMITIV_TRAINER_IMPL_H_
