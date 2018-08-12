#include <primitiv/config.h>

#include <chrono>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <primitiv/core/error.h>
#include <primitiv/devices/naive/device.h>
#include <primitiv/core/shape.h>
#include <primitiv/core/tensor.h>

#include <test_utils.h>

using std::vector;
using test_utils::vector_match;
using test_utils::vector_near;

namespace primitiv {

class NaiveDeviceTest : public testing::Test {};

TEST_F(NaiveDeviceTest, CheckDeviceType) {
  devices::Naive dev;
  EXPECT_EQ(DeviceType::NAIVE, dev.type());
}

TEST_F(NaiveDeviceTest, CheckNewDelete) {
  {
    devices::Naive dev;
    {
      // 1 value
      Tensor x1 = dev.new_tensor_by_constant(Shape(), 0);
      // 256 values
      Tensor x2 = dev.new_tensor_by_constant(Shape({16, 16}), 0);
      // 65536 values
      Tensor x3 = dev.new_tensor_by_constant(Shape({16, 16, 16}, 16), 0);
    }
    // All tensors are already deleted before arriving here.
  }
  SUCCEED();
}

TEST_F(NaiveDeviceTest, CheckDanglingTensor) {
  {
    Tensor x1;
    {
      devices::Naive dev;
      x1 = dev.new_tensor_by_constant(Shape(), 0);
    }
    // x1 still has valid object,
    // but there is no guarantee that the memory is alive.
    // Our implementation only guarantees the safety to delete Tensors anytime.
  }
  SUCCEED();
}

#ifdef PRIMITIV_BUILD_TESTS_PROBABILISTIC
TEST_F(NaiveDeviceTest, CheckRandomBernoulli) {
  vector<vector<float>> history;
  for (std::uint32_t i = 0; i < 10; ++i) {
    devices::Naive dev;
    const Tensor x = dev.random_bernoulli(Shape({3, 3}, 3), 0.3);
    const vector<float> x_val = x.to_vector();

    std::cout << "Epoch " << i << ':';
    for (float x_i : x_val) {
      std::cout << ' ' << x_i;
    }
    std::cout << std::endl;

    for (const vector<float> &h_val : history) {
      EXPECT_FALSE(vector_match(x_val, h_val));
    }
    history.emplace_back(x_val);

    // Wait for updating the device randomizer.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}
#endif  // PRIMITIV_BUILD_TESTS_PROBABILISTIC

TEST_F(NaiveDeviceTest, CheckRandomBernoulliWithSeed) {
  const vector<float> expected {
    0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
  };
  devices::Naive dev(12345);
  const Tensor x = dev.random_bernoulli(Shape({4, 4}, 4), 0.3);
  EXPECT_TRUE(vector_match(expected, x.to_vector()));
}

#ifdef PRIMITIV_BUILD_TESTS_PROBABILISTIC
TEST_F(NaiveDeviceTest, CheckRandomUniform) {
  vector<vector<float>> history;
  for (std::uint32_t i = 0; i < 10; ++i) {
    devices::Naive dev;
    const Tensor x = dev.random_uniform(Shape({2, 2}, 2), -9, 9);
    const vector<float> x_val = x.to_vector();

    std::cout << "Epoch " << i << ':';
    for (float x_i : x_val) {
      std::cout << ' ' << x_i;
    }
    std::cout << std::endl;

    for (const vector<float> &h_val : history) {
      EXPECT_FALSE(vector_match(x_val, h_val));
    }
    history.emplace_back(x_val);

    // Wait for updating the device randomizer.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}
#endif  // PRIMITIV_BUILD_TESTS_PROBABILISTIC

TEST_F(NaiveDeviceTest, CheckRandomUniformWithSeed) {
  const vector<float> expected {
    7.7330894e+00, 7.0227852e+00, -3.3052402e+00, -6.6472688e+00,
    -5.6894612e+00, -8.2843294e+00, -5.3179150e+00, 5.8758497e+00,
  };
  devices::Naive dev(12345);
  const Tensor x = dev.random_uniform(Shape({2, 2}, 2), -9, 9);
  EXPECT_TRUE(vector_match(expected, x.to_vector()));
}

#ifdef PRIMITIV_BUILD_TESTS_PROBABILISTIC
TEST_F(NaiveDeviceTest, CheckRandomNormal) {
  vector<vector<float>> history;
  for (std::uint32_t i = 0; i < 10; ++i) {
    devices::Naive dev;
    const Tensor x = dev.random_normal(Shape({2, 2}, 2), 1, 3);
    const vector<float> x_val = x.to_vector();

    std::cout << "Epoch " << i << ':';
    for (float x_i : x_val) {
      std::cout << ' ' << x_i;
    }
    std::cout << std::endl;

    for (const vector<float> &h_val : history) {
      EXPECT_FALSE(vector_match(x_val, h_val));
    }
    history.emplace_back(x_val);

    // Wait for updating the device randomizer.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}
#endif  // PRIMITIV_BUILD_TESTS_PROBABILISTIC

TEST_F(NaiveDeviceTest, CheckRandomNormalWithSeed) {
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
  devices::Naive dev(12345);
  const Tensor x = dev.random_normal(Shape({2, 2}, 2), 1, 3);
#ifdef PRIMITIV_MAYBE_FPMATH_X87
  EXPECT_TRUE(vector_near(expected, x.to_vector(), 1e-6));
#else
  EXPECT_TRUE(vector_match(expected, x.to_vector()));
#endif
}

#ifdef PRIMITIV_BUILD_TESTS_PROBABILISTIC
TEST_F(NaiveDeviceTest, CheckRandomLogNormal) {
  vector<vector<float>> history;
  for (std::uint32_t i = 0; i < 10; ++i) {
    devices::Naive dev;
    const Tensor x = dev.random_log_normal(Shape({2, 2}, 2), 1, 3);
    const vector<float> x_val = x.to_vector();

    std::cout << "Epoch " << i << ':';
    for (float x_i : x_val) {
      std::cout << ' ' << x_i;
    }
    std::cout << std::endl;

    for (const vector<float> &h_val : history) {
      EXPECT_FALSE(vector_match(x_val, h_val));
    }
    history.emplace_back(x_val);

    // Wait for updating the device randomizer.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}
#endif  // PRIMITIV_BUILD_TESTS_PROBABILISTIC

TEST_F(NaiveDeviceTest, CheckRandomLogNormalWithSeed) {
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
  devices::Naive dev(12345);
  const Tensor x = dev.random_log_normal(Shape({2, 2}, 2), 1, 3);
#ifdef PRIMITIV_MAYBE_FPMATH_X87
  EXPECT_TRUE(vector_near(expected, x.to_vector(), 1e-4));
#else
  EXPECT_TRUE(vector_match(expected, x.to_vector()));
#endif
}

}  // namespace primitiv
