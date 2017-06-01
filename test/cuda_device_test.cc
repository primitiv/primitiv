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

TEST_F(CUDADeviceTest, CheckInvalidInit) {
  EXPECT_THROW(CUDADevice dev(12345678), Error);
}

TEST_F(CUDADeviceTest, CheckNewDelete) {
  {
    CUDADevice dev(0);
    Tensor x1 = dev.new_tensor(Shape()); // 1 value
    Tensor x2 = dev.new_tensor(Shape {16, 16}); // 256 values
    Tensor x3 = dev.new_tensor(Shape({16, 16, 16}, 16)); // 65536 values
    // According to the C++ standard, local values are destroyed in the order:
    // x3 -> x2 -> x1 -> dev.
    // Then `dev` has no remaining memories.
  }
  SUCCEED();
}

/*
 * TODO(odashi): the death test requires a single-thread program,
 *               but CUDA behaves on multi-threads.
TEST_F(CUDADeviceTest, CheckInvalidNewDelete) {
  EXPECT_DEATH({
    Tensor x0;
    CUDADevice dev(0);
    x0 = dev.new_tensor(Shape());
    // Local values are destroyed in the order: dev -> x0.
    // `x0` still have a memory when destroying `dev` and the process will
    // abort.
  }, "");
}
*/

TEST_F(CUDADeviceTest, CheckSetValuesByConstant) {
  CUDADevice dev(0);
  {
    Tensor x = dev.new_tensor(Shape({2, 2}, 2), 42);
    EXPECT_TRUE(vector_match(vector<float>(8, 42), x.to_vector()));
  }
  {
    Tensor x = dev.new_tensor(Shape({2, 2}, 2));
    x.reset(42);
    EXPECT_TRUE(vector_match(vector<float>(8, 42), x.to_vector()));
  }
}

TEST_F(CUDADeviceTest, CheckSetValuesByVector) {
  CUDADevice dev(0);
  {
    const vector<float> data {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor x = dev.new_tensor(Shape({2, 2}, 2), data);
    EXPECT_TRUE(vector_match(data, x.to_vector()));
  }
  {
    const vector<float> data {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor x = dev.new_tensor(Shape({2, 2}, 2));
    x.reset(data);
    EXPECT_TRUE(vector_match(data, x.to_vector()));
  }
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
