#include <config.h>

#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/shape.h>
#include <primitiv/tensor.h>
#include <test_utils.h>

using std::vector;
using test_utils::vector_match;

namespace primitiv {

class CPUDeviceTest : public testing::Test {};

TEST_F(CPUDeviceTest, CheckNewDelete) {
  {
    CPUDevice dev;
    Tensor x1 = dev.new_tensor(Shape()); // 1 value
    Tensor x2 = dev.new_tensor(Shape {16, 16}); // 256 values
    Tensor x3 = dev.new_tensor(Shape({16, 16, 16}, 16)); // 65536 values
    // According to the C++ standard, local values are destroyed in the order:
    // x3 -> x2 -> x1 -> dev.
    // Then `dev` has no remaining memories.
  }
  SUCCEED();
}

TEST_F(CPUDeviceTest, CheckInvalidNewDelete) {
  EXPECT_DEATH({
    Tensor x0;
    CPUDevice dev;
    x0 = dev.new_tensor(Shape());
    // Local values are destroyed in the order: dev -> x0.
    // `x0` still have a memory when destroying `dev` and the process will
    // abort.
  }, "");
}

TEST_F(CPUDeviceTest, CheckSetValuesByConstant) {
  CPUDevice dev;
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

TEST_F(CPUDeviceTest, CheckSetValuesByArray) {
  CPUDevice dev;
  {
    const float data[] {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor x = dev.new_tensor_by_array(Shape({2, 2}, 2), data);
    EXPECT_TRUE(
        vector_match(vector<float> {1, 2, 3, 4, 5, 6, 7, 8}, x.to_vector()));
  }
  {
    const float data[] {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor x = dev.new_tensor(Shape({2, 2}, 2));
    x.reset_by_array(data);
    EXPECT_TRUE(
        vector_match(vector<float> {1, 2, 3, 4, 5, 6, 7, 8}, x.to_vector()));
  }
}

TEST_F(CPUDeviceTest, CheckSetValuesByVector) {
  CPUDevice dev;
  {
    const vector<float> data {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor x = dev.new_tensor_by_vector(Shape({2, 2}, 2), data);
    EXPECT_TRUE(vector_match(data, x.to_vector()));
  }
  {
    const vector<float> data {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor x = dev.new_tensor(Shape({2, 2}, 2));
    x.reset_by_vector(data);
    EXPECT_TRUE(vector_match(data, x.to_vector()));
  }
}

TEST_F(CPUDeviceTest, CheckRandomBernoulli) {
  vector<vector<float>> history;
  for (unsigned i = 0; i < 10; ++i) {
    CPUDevice dev;
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

TEST_F(CPUDeviceTest, CheckRandomBernoulliWithSeed) {
  const vector<float> expected {
    0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
  };
  CPUDevice dev(12345);
  const Tensor x = dev.random_bernoulli(Shape({4, 4}, 4), 0.3);
  EXPECT_TRUE(vector_match(expected, x.to_vector()));
}

TEST_F(CPUDeviceTest, CheckRandomUniform) {
  vector<vector<float>> history;
  for (unsigned i = 0; i < 10; ++i) {
    CPUDevice dev;
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

TEST_F(CPUDeviceTest, CheckRandomUniformWithSeed) {
  const vector<float> expected {
    7.7330894e+00, 7.0227852e+00, -3.3052402e+00, -6.6472688e+00,
    -5.6894612e+00, -8.2843294e+00, -5.3179150e+00, 5.8758497e+00,
  };
  CPUDevice dev(12345);
  const Tensor x = dev.random_uniform(Shape({2, 2}, 2), -9, 9);
  EXPECT_TRUE(vector_match(expected, x.to_vector()));
}

TEST_F(CPUDeviceTest, CheckRandomNormal) {
  vector<vector<float>> history;
  for (unsigned i = 0; i < 10; ++i) {
    CPUDevice dev;
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

TEST_F(CPUDeviceTest, CheckRandomNormalWithSeed) {
  const vector<float> expected {
    -1.3574908e+00, -1.7222166e-01, 2.5865970e+00, -4.3594337e-01,
    4.5383353e+00, 8.4703674e+00, 2.5535507e+00, 1.3252910e+00,
  };
  CPUDevice dev(12345);
  const Tensor x = dev.random_normal(Shape({2, 2}, 2), 1, 3);
  EXPECT_TRUE(vector_match(expected, x.to_vector()));
}

}  // namespace primitiv
