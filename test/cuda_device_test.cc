#include <primitiv/config.h>

#include <chrono>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/core/error.h>
#include <primitiv/core/shape.h>
#include <primitiv/core/tensor.h>

#include <test_utils.h>

using std::vector;
using test_utils::vector_match;

namespace primitiv {

class CUDADeviceTest : public testing::Test {
protected:
  vector<std::uint32_t> dev_ids;

  void SetUp() override {
    const std::uint32_t num_devs = devices::CUDA::num_devices();
    for (std::uint32_t dev_id = 0; dev_id < num_devs; ++dev_id) {
      if (devices::CUDA::check_support(dev_id)) {
        dev_ids.emplace_back(dev_id);
      }
    }
    if (dev_ids.empty()) {
      std::cerr << "No available CUDA devices." << std::endl;
    }
  }
};

TEST_F(CUDADeviceTest, CheckDeviceType) {
  for (std::uint32_t dev_id : dev_ids) {
    devices::CUDA dev(dev_id);
    EXPECT_EQ(DeviceType::CUDA, dev.type());
  }
}

TEST_F(CUDADeviceTest, CheckInvalidInit) {
  EXPECT_THROW(devices::CUDA(devices::CUDA::num_devices()), Error);
  EXPECT_THROW(devices::CUDA(12345678), Error);
}

TEST_F(CUDADeviceTest, CheckNewDelete) {
  for (std::uint32_t dev_id : dev_ids) {
    devices::CUDA dev(dev_id);
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

TEST_F(CUDADeviceTest, CheckDanglingTensor) {
  for (std::uint32_t dev_id : dev_ids) {
    Tensor x1;
    {
      devices::CUDA dev(dev_id);
      x1 = dev.new_tensor_by_constant(Shape(), 0);
    }
    // x1 still has valid object,
    // but there is no guarantee that the memory is alive.
    // Our implementation only guarantees the safety to delete Tensors anytime.
  }
  SUCCEED();
}

#ifdef PRIMITIV_BUILD_TESTS_PROBABILISTIC
TEST_F(CUDADeviceTest, CheckRandomBernoulli) {
  vector<vector<float>> history;
  for (std::uint32_t dev_id : dev_ids) {
    for (std::uint32_t i = 0; i < 10; ++i) {
      devices::CUDA dev(dev_id);
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
}
#endif  // PRIMITIV_BUILD_TESTS_PROBABILISTIC

TEST_F(CUDADeviceTest, CheckRandomBernoulliWithSeed) {
  const vector<float> expected {
    1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
  };
  for (std::uint32_t dev_id : dev_ids) {
    devices::CUDA dev(dev_id, 12345);
    const Tensor x = dev.random_bernoulli(Shape({4, 4}, 4), 0.3);
    EXPECT_TRUE(vector_match(expected, x.to_vector()));
  }
}

#ifdef PRIMITIV_BUILD_TESTS_PROBABILISTIC
TEST_F(CUDADeviceTest, CheckRandomUniform) {
  vector<vector<float>> history;
  for (std::uint32_t dev_id : dev_ids) {
    for (std::uint32_t i = 0; i < 10; ++i) {
      devices::CUDA dev(dev_id);
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
}
#endif  // PRIMITIV_BUILD_TESTS_PROBABILISTIC

TEST_F(CUDADeviceTest, CheckRandomUniformWithSeed) {
  const vector<float> expected {
    -3.6198268e+00, 4.1064610e+00, -6.9007745e+00, 8.5519943e+00,
    -7.7016129e+00, -4.6067810e+00, 8.7706423e+00, -4.9437490e+00,
  };
  for (std::uint32_t dev_id : dev_ids) {
    devices::CUDA dev(dev_id, 12345);
    const Tensor x = dev.random_uniform(Shape({2, 2}, 2), -9, 9);
    EXPECT_TRUE(vector_match(expected, x.to_vector()));
  }
}

#ifdef PRIMITIV_BUILD_TESTS_PROBABILISTIC
TEST_F(CUDADeviceTest, CheckRandomNormal) {
  vector<vector<float>> history;
  for (std::uint32_t dev_id : dev_ids) {
    for (std::uint32_t i = 0; i < 10; ++i) {
      devices::CUDA dev(dev_id);
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
}
#endif  // PRIMITIV_BUILD_TESTS_PROBABILISTIC

TEST_F(CUDADeviceTest, CheckRandomNormalWithSeed) {
  const vector<float> expected {
    4.1702256e+00, -2.4186814e+00, 1.5060894e+00, -1.3355234e+00,
    -5.0218196e+00, -5.5439359e-01, 5.8913720e-01, 1.5337296e+00,
  };
  for (std::uint32_t dev_id : dev_ids) {
    devices::CUDA dev(dev_id, 12345);
    const Tensor x = dev.random_normal(Shape({2, 2}, 2), 1, 3);
    EXPECT_TRUE(vector_match(expected, x.to_vector()));
  }
}

#ifdef PRIMITIV_BUILD_TESTS_PROBABILISTIC
TEST_F(CUDADeviceTest, CheckRandomLogNormal) {
  vector<vector<float>> history;
  for (std::uint32_t dev_id : dev_ids) {
    for (std::uint32_t i = 0; i < 10; ++i) {
      devices::CUDA dev(dev_id);
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
}
#endif  // PRIMITIV_BUILD_TESTS_PROBABILISTIC

TEST_F(CUDADeviceTest, CheckRandomLogNormalWithSeed) {
  const vector<float> expected {
    6.4730049e+01, 8.9038946e-02, 4.5090632e+00, 2.6302049e-01,
    6.5925200e-03, 5.7442045e-01, 1.8024327e+00, 4.6354327e+00,
  };
  for (std::uint32_t dev_id : dev_ids) {
    devices::CUDA dev(dev_id, 12345);
    const Tensor x = dev.random_log_normal(Shape({2, 2}, 2), 1, 3);
    EXPECT_TRUE(vector_match(expected, x.to_vector()));
  }
}

}  // namespace primitiv
