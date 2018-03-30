#ifndef PRIMITIV_RANDOM_H_
#define PRIMITIV_RANDOM_H_

#include <cstddef>
#include <limits>
#include <random>
#include <primitiv/mixins.h>

namespace primitiv {

/**
 * Default randomizer for any devices.
 */
class DefaultRandomizer : mixins::Nonmovable<DefaultRandomizer> {
  std::mt19937 rng_;

public:
  /**
   * Creates a randomizer object using environment seeds.
   */
  DefaultRandomizer() : rng_(std::random_device()()) {}

  /**
   * Creates a randomizer object using a user seed.
   * @param seed Seed value of the randomizer.
   */
  explicit DefaultRandomizer(std::uint32_t seed) : rng_(seed) {}

  /**
   * Fill an array using a Bernoulli distribution.
   * @param p Probability with witch the variable becomes 1.
   * @param size Length of the array `data`.
   * @param data Pointer of the array in which results are stored.
   */
  void fill_bernoulli(float p, std::size_t size, float *data) {
    std::bernoulli_distribution dist(p);
    for (std::size_t i = 0; i < size; ++i) {
      data[i] = dist(rng_);
    }
  }

  /**
   * Fill an array using a uniform distribution.
   * @param lower Lower bound of the distribution.
   * @param upper Upper bound of the distribution.
   * @param size Length of the array `data`.
   * @param data Pointer of the array in which results are stored.
   * @remarks Range of the resulting sequence is (lower, upper].
   */
  void fill_uniform(float lower, float upper, std::size_t size, float *data) {
    // NOTE(vbkaisetsu):
    // std::uniform_real_distribution generates compiler dependent results.
    // Our implementation guarantees results of the uniform distribution.
    static_assert(std::numeric_limits<float>::digits == 24, "");
    const std::uint32_t h = (rng_.max() - rng_.min()) / 2 + rng_.min();
    for (std::size_t j = 0; j < size; ++j) {
      float r = 0.f;
      float s = 1.f;
      for (std::uint32_t e = 0; e < std::numeric_limits<float>::digits; ++e) {
        s *= .5f;
        if (rng_() > h) {
          r += s;
        }
      }
      data[j] = -(r * (upper - lower) - upper);
    }
  }

  /**
   * Fill an array using a normal distribution.
   * @param mean Mean of the distribution.
   * @param sd Standard deviation of the distribution.
   * @param size Length of the array `data`.
   * @param data Pointer of the array in which results are stored.
   */
  void fill_normal(float mean, float sd, std::size_t size, float *data) {
    std::normal_distribution<float> dist(mean, sd);
    for (std::size_t i = 0; i < size; ++i) {
      data[i] = dist(rng_);
    }
  }

  /**
   * Fill an array using a log-normal distribution.
   * @param mean Mean of the corresponding normal distribution.
   * @param sd Standard deviation of the corresponding normal distribution.
   * @param size Length of the array `data`.
   * @param data Pointer of the array in which results are stored.
   */
  void fill_log_normal(float mean, float sd, std::size_t size, float *data) {
    std::lognormal_distribution<float> dist(mean, sd);
    for (std::size_t i = 0; i < size; ++i) {
      data[i] = dist(rng_);
    }
  }
};

}  // namespace primitiv

#endif  // PRIMITIV_RANDOM_H_
