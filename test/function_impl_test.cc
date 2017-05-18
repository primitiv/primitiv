#include <config.h>

#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/function_impl.h>
#include <test_utils.h>

using std::vector;

namespace primitiv {
namespace functions {

class FunctionImplTest_0Arg : public testing::Test {
protected:
  CPUDevice dev;
  vector<const Shape *> arg_shapes;
  vector<const Tensor *> arg_values;
  vector<Tensor *> arg_grads;
};

TEST_F(FunctionImplTest_0Arg, CheckInput) {
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4};
  const Input node(ret_shape, &dev, ret_data);
  const Shape cur_shape = node.forward_shape(arg_shapes);
  const Tensor cur_value = node.forward(arg_values);
  const Tensor cur_grad = dev.constant(ret_shape, 1);
  // backward() has no effect.
  EXPECT_NO_THROW(node.backward(cur_value, cur_grad, arg_values, arg_grads));
  EXPECT_EQ("Input", node.name());
  EXPECT_EQ(ret_shape, cur_shape);
  EXPECT_TRUE(test_utils::vector_match(ret_data, cur_value.to_vector()));
}

class FunctionImplTest_1Arg : public testing::Test {
protected:
  virtual void SetUp() override {
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_values.emplace_back(new Tensor(
        *arg_shapes[0], &dev,
        vector<float> {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4}));
    arg_grads.emplace_back(new Tensor(
        *arg_shapes[0], &dev, vector<float>(arg_shapes[0]->size())));
  }

  virtual void TearDown() override {
    for (const Shape *x : arg_shapes) delete x;
    for (const Tensor *x : arg_values) delete x;
    for (Tensor *x : arg_grads) delete x;
  }

  CPUDevice dev;
  vector<const Shape *> arg_shapes;
  vector<const Tensor *> arg_values;
  vector<Tensor *> arg_grads;
};

#define TEST_ALL(name_) { \
  const Shape cur_shape = node.forward_shape(arg_shapes); \
  const Tensor cur_value = node.forward(arg_values); \
  const Tensor cur_grad = dev.constant(ret_shape, 1); \
  node.backward(cur_value, cur_grad, arg_values, arg_grads); \
  EXPECT_EQ(#name_, node.name()); \
  EXPECT_EQ(ret_shape, cur_shape); \
  EXPECT_TRUE(test_utils::vector_match(ret_data, cur_value.to_vector())); \
  EXPECT_TRUE(test_utils::vector_match(bw_grad, arg_grads[0]->to_vector())); \
}

TEST_F(FunctionImplTest_1Arg, CheckAddConst) {
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {4, 5, 6, 7, 3, 3, 3, 3, 2, 1, 0, -1};
  const vector<float> bw_grad(arg_shapes[0]->size(), 1);
  const AddConst node(3);
  TEST_ALL(AddConst);
}

#undef TEST_ALL

class FunctionImplTest_2Args : public testing::Test {
protected:
  virtual void SetUp() override {
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_values.emplace_back(new Tensor(
        *arg_shapes[0], &dev,
        vector<float> {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4}));
    arg_values.emplace_back(new Tensor(
        *arg_shapes[1], &dev,
        vector<float> {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}));
    arg_grads.emplace_back(new Tensor(
        *arg_shapes[0], &dev, vector<float>(arg_shapes[0]->size())));
    arg_grads.emplace_back(new Tensor(
        *arg_shapes[1], &dev, vector<float>(arg_shapes[1]->size())));
  }

  virtual void TearDown() override {
    for (const Shape *x : arg_shapes) delete x;
    for (const Tensor *x : arg_values) delete x;
    for (Tensor *x : arg_grads) delete x;
  }

  CPUDevice dev;
  vector<const Shape *> arg_shapes;
  vector<const Tensor *> arg_values;
  vector<Tensor *> arg_grads;
};

#define TEST_ALL(name_) { \
  const Shape cur_shape = node.forward_shape(arg_shapes); \
  const Tensor cur_value = node.forward(arg_values); \
  const Tensor cur_grad = dev.constant(ret_shape, 1); \
  node.backward(cur_value, cur_grad, arg_values, arg_grads); \
  EXPECT_EQ(#name_, node.name()); \
  EXPECT_EQ(ret_shape, cur_shape); \
  EXPECT_TRUE(test_utils::vector_match(ret_data, cur_value.to_vector())); \
  EXPECT_TRUE( \
      test_utils::vector_match(bw_grads[0], arg_grads[0]->to_vector())); \
  EXPECT_TRUE( \
      test_utils::vector_match(bw_grads[1], arg_grads[1]->to_vector())); \
}

TEST_F(FunctionImplTest_2Args, CheckAdd) {
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {2, 3, 4, 5, 2, 2, 2, 2, 2, 1, 0, -1};
  const vector<vector<float>> bw_grads {
    vector<float>(arg_shapes[0]->size(), 1),
    vector<float>(arg_shapes[1]->size(), 1),
  };
  const Add node;
  TEST_ALL(Add);
}

TEST_F(FunctionImplTest_2Args, CheckSubtract) {
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {0, 1, 2, 3, -2, -2, -2, -2, -4, -5, -6, -7};
  const vector<vector<float>> bw_grads {
    vector<float>(arg_shapes[0]->size(), 1),
    vector<float>(arg_shapes[1]->size(), -1),
  };
  const Subtract node;
  TEST_ALL(Subtract);
}

TEST_F(FunctionImplTest_2Args, CheckMultiply) {
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {1, 2, 3, 4, 0, 0, 0, 0, -3, -6, -9, -12};
  const vector<vector<float>> bw_grads {
    vector<float> {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3},
    vector<float> {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4},
  };
  const Multiply node;
  TEST_ALL(Multiply);
}

#undef TEST_ALL

}  // namespace functions
}  // namespace primitiv
