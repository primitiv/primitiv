#include <primitiv/config.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>

#include <gtest/gtest.h>

#include <primitiv/core/functions.h>
#include <primitiv/core/graph.h>
#include <primitiv/devices/naive/device.h>

#include <test_utils.h>

using std::vector;
using test_utils::vector_match;

namespace primitiv {

class NodeTest : public testing::Test {
protected:
  devices::Naive dev;
  Graph g;

  void SetUp() override {
    Device::set_default(dev);
    Graph::set_default(g);
  }
};

TEST_F(NodeTest, CheckArgMaxDims) {
  const vector<float> data = {
    0, 1, 2, 6, 7, 8, 3, 4, 5, -3, -4, -5, 0, -1, -2, -6, -7, -8,
  };
  const vector<vector<std::uint32_t>> expected = {
    {2, 2, 2, 0, 0, 0},
    {1, 1, 1, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  };
  const Node a = functions::input<Node>(Shape({3, 3}, 2), data);
  for (const std::uint32_t i : {0u, 1u, 2u}) {
    EXPECT_TRUE(vector_match(expected[i], a.argmax(i)));
  }
}

TEST_F(NodeTest, CheckArgMaxLarge) {
  std::mt19937 rng;
  const vector<std::uint32_t> ns {
    1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 65535, 65536, 65537,
  };
  for (const std::uint32_t n : ns) {
    vector<float> data(n);
    std::iota(begin(data), end(data), 0);
    std::shuffle(begin(data), end(data), rng);
    const auto it = std::find(begin(data), end(data), n - 1);
    const std::uint32_t pos = std::distance(begin(data), it);
    const Node a = functions::input<Node>({n}, data);
    EXPECT_EQ(pos, a.argmax(0)[0]);
  }
}

TEST_F(NodeTest, CheckArgMinDims) {
  const vector<float> data = {
    3, 4, 5, 0, 1, 2, 6, 7, 8, 0, -1, -2, -6, -7, -8, -3, -4, -5,
  };
  const vector<vector<std::uint32_t>> expected = {
    {0, 0, 0, 2, 2, 2},
    {1, 1, 1, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  };
  const Node a = functions::input<Node>(Shape({3, 3}, 2), data);
  for (const std::uint32_t i : {0u, 1u, 2u}) {
    EXPECT_TRUE(vector_match(expected[i], a.argmin(i)));
  }
}

TEST_F(NodeTest, CheckArgMinLarge) {
  std::mt19937 rng;
  const vector<std::uint32_t> ns {
    1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 65535, 65536, 65537,
  };
  for (const std::uint32_t n : ns) {
    vector<float> data(n);
    std::iota(begin(data), end(data), 0);
    std::shuffle(begin(data), end(data), rng);
    const auto it = std::find(begin(data), end(data), 0);
    const std::uint32_t pos = std::distance(begin(data), it);
    const Node a = functions::input<Node>({n}, data);
    EXPECT_EQ(pos, a.argmin(0)[0]);
  }
}

}  // namespace primitiv
