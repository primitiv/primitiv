#include <primitiv/config.h>

#include <vector>

#include <gtest/gtest.h>

#include <primitiv/core/random.h>

#include <test_utils.h>

using std::vector;
using test_utils::vector_match;
using test_utils::vector_near;

namespace primitiv {

class DefaultRandomizerTest : public testing::Test {
protected:
  DefaultRandomizer randomizer_;

  DefaultRandomizerTest() : randomizer_(12345) {}
};

TEST_F(DefaultRandomizerTest, CheckFillBernoulli) {
  const vector<float> expected {
    0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
  };

  const std::size_t size = expected.size();
  vector<float> observed(size, -1e10);
  randomizer_.fill_bernoulli(0.3, size, observed.data());
  EXPECT_TRUE(vector_match(expected, observed));
}

TEST_F(DefaultRandomizerTest, CheckFillUniform) {
  const vector<float> expected {
    7.7330894e+00, 7.0227852e+00, -3.3052402e+00, -6.6472688e+00,
    -5.6894612e+00, -8.2843294e+00, -5.3179150e+00, 5.8758497e+00,
  };

  const std::size_t size = expected.size();
  vector<float> observed(size, -1e10);
  randomizer_.fill_uniform(-9, 9, size, observed.data());
  EXPECT_TRUE(vector_match(expected, observed));
}

TEST_F(DefaultRandomizerTest, CheckFillNormal) {
#ifdef __GLIBCXX__
  const vector<float> expected {
    -1.3574908e+00, -1.7222166e-01, 2.5865970e+00, -4.3594337e-01,
    4.5383353e+00, 8.4703674e+00, 2.5535507e+00, 1.3252910e+00,
  };
#elif defined _LIBCPP_VERSION
  const vector<float> expected {
    -1.7222166e-01, -1.3574908e+00, -4.3594337e-01, 2.5865970e+00,
    8.4703674e+00, 4.5383353e+00, 1.3252910e+00, 2.5535507e+00,
  };
#else
  const vector<float> expected {};
  std::cerr << "Unknown C++ library. Expected results can't be defined." << std::endl;
#endif

  const std::size_t size = expected.size();
  vector<float> observed(size, -1e10);
  randomizer_.fill_normal(1, 3, size, observed.data());
#ifdef PRIMITIV_MAYBE_FPMATH_X87
  EXPECT_TRUE(vector_near(expected, observed, 1e-6));
#else
  EXPECT_TRUE(vector_match(expected, observed));
#endif
}

TEST_F(DefaultRandomizerTest, CheckFillLogNormal) {
#ifdef __GLIBCXX__
  const vector<float> expected {
    2.5730559e-01, 8.4179258e-01, 1.3284487e+01, 6.4665437e-01,
    9.3534966e+01, 4.7712681e+03, 1.2852659e+01, 3.7632804e+00,
  };
#elif defined _LIBCPP_VERSION
  const vector<float> expected {
    8.4179258e-01, 2.5730559e-01, 6.4665437e-01, 1.3284487e+01,
    4.7712681e+03, 9.3534966e+01, 3.7632804e+00, 1.2852659e+01,
  };
#else
  const vector<float> expected {};
  std::cerr << "Unknown C++ library. Expected results can't be defined." << std::endl;
#endif

  const std::size_t size = expected.size();
  vector<float> observed(size, -1e10);
  randomizer_.fill_log_normal(1, 3, size, observed.data());
#ifdef PRIMITIV_MAYBE_FPMATH_X87
  EXPECT_TRUE(vector_near(expected, observed, 1e-4));
#else
  EXPECT_TRUE(vector_match(expected, observed));
#endif
}

}  // namespace primitiv
