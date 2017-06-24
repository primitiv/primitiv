#include <config.h>

#include <chrono>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cuda_device.h>
#include <primitiv/error.h>
#include <primitiv/shape.h>
#include <primitiv/tensor.h>
#include <test_utils.h>

using std::vector;
using test_utils::vector_match;

namespace primitiv {

class CUDADeviceTest : public testing::Test {};

TEST_F(CUDADeviceTest, CheckDeviceType) {
  CUDADevice dev(0);
  EXPECT_EQ(Device::DEVICE_TYPE_CUDA, dev.type());
}

TEST_F(CUDADeviceTest, CheckInvalidInit) {
  // We might not have millions of GPUs in one host.
  EXPECT_THROW(CUDADevice dev(12345678), Error);
}

TEST_F(CUDADeviceTest, CheckRandomBernoulli) {
  vector<vector<float>> history;
  for (unsigned i = 0; i < 10; ++i) {
    CUDADevice dev(0);
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

TEST_F(CUDADeviceTest, CheckRandomBernoulliWithSeed) {
  const vector<float> expected {
    1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
  };
  CUDADevice dev(0, 12345);
  const Tensor x = dev.random_bernoulli(Shape({4, 4}, 4), 0.3);
  EXPECT_TRUE(vector_match(expected, x.to_vector()));
}

TEST_F(CUDADeviceTest, CheckRandomUniform) {
  vector<vector<float>> history;
  for (unsigned i = 0; i < 10; ++i) {
    CUDADevice dev(0);
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

TEST_F(CUDADeviceTest, CheckRandomUniformWithSeed) {
  const vector<float> expected {
    -3.6198268e+00, 4.1064610e+00, -6.9007745e+00, 8.5519943e+00,
    -7.7016129e+00, -4.6067810e+00, 8.7706423e+00, -4.9437490e+00,
  };
  CUDADevice dev(0, 12345);
  const Tensor x = dev.random_uniform(Shape({2, 2}, 2), -9, 9);
  EXPECT_TRUE(vector_match(expected, x.to_vector()));
}

TEST_F(CUDADeviceTest, CheckRandomNormal) {
  vector<vector<float>> history;
  for (unsigned i = 0; i < 10; ++i) {
    CUDADevice dev(0);
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

TEST_F(CUDADeviceTest, CheckRandomNormalWithSeed) {
  const vector<float> expected {
    4.1702256e+00, -2.4186814e+00, 1.5060894e+00, -1.3355234e+00,
    -5.0218196e+00, -5.5439359e-01, 5.8913720e-01, 1.5337296e+00,
  };
  CUDADevice dev(0, 12345);
  const Tensor x = dev.random_normal(Shape({2, 2}, 2), 1, 3);
  EXPECT_TRUE(vector_match(expected, x.to_vector()));
}

}  // namespace primitiv
