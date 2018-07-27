#include <primitiv/config.h>

#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include <primitiv/c/devices/naive/device.h>
#include <primitiv/c/functions.h>
#include <primitiv/c/graph.h>
#include <primitiv/c/parameter.h>
#include <primitiv/c/status.h>
#include <primitiv/c/tensor.h>

#include <test_utils.h>

using std::vector;
using test_utils::array_match;

namespace primitiv {
namespace c {

class CGraphTest : public testing::Test {
  void SetUp() override {
    ::primitivCreateNaiveDevice(&dev);
    ::primitivCreateNaiveDevice(&dev2);
  }
  void TearDown() override {
    ::primitivDeleteDevice(dev);
    ::primitivDeleteDevice(dev2);
  }
 protected:
  ::primitivDevice_t *dev;
  ::primitivDevice_t *dev2;
};

TEST_F(CGraphTest, CheckDefault) {
  ::primitivResetStatus();
  ::primitivGraph_t *graph;
  EXPECT_EQ(PRIMITIV_C_ERROR,
            ::primitivGetDefaultGraph(&graph));
  {
    ::primitivGraph_t *g1;
    ASSERT_EQ(PRIMITIV_C_OK, ::primitivCreateGraph(&g1));
    ::primitivSetDefaultGraph(g1);
    ::primitivGetDefaultGraph(&graph);
    EXPECT_EQ(g1, graph);
    {
      ::primitivGraph_t *g2;
      ASSERT_EQ(PRIMITIV_C_OK, ::primitivCreateGraph(&g2));
      ::primitivSetDefaultGraph(g2);
      ::primitivGetDefaultGraph(&graph);
      EXPECT_EQ(g2, graph);
      ::primitivDeleteGraph(g2);
    }
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitivGetDefaultGraph(&graph));
    ::primitivGraph_t *g3;
    ASSERT_EQ(PRIMITIV_C_OK, ::primitivCreateGraph(&g3));
    ::primitivSetDefaultGraph(g3);
    ::primitivGetDefaultGraph(&graph);
    EXPECT_EQ(g3, graph);
    ::primitivDeleteGraph(g1);
    ::primitivDeleteGraph(g3);
  }
  EXPECT_EQ(PRIMITIV_C_ERROR,
            ::primitivGetDefaultGraph(&graph));
}

TEST_F(CGraphTest, CheckInvalidNode) {
  ::primitivResetStatus();
  ::primitivNode_t *node;
  ASSERT_EQ(PRIMITIV_C_OK, ::primitivCreateNode(&node));
  PRIMITIV_C_BOOL valid;
  ::primitivIsValidNode(node, &valid);
  EXPECT_FALSE(valid);
  ::primitivGraph_t *graph;
  EXPECT_EQ(PRIMITIV_C_ERROR,
            ::primitivGetGraphFromNode(node, &graph));
  uint32_t id;
  EXPECT_EQ(PRIMITIV_C_ERROR,
            ::primitivGetNodeOperatorId(node, &id));
  EXPECT_EQ(PRIMITIV_C_ERROR,
            ::primitivGetNodeValueId(node, &id));
  ::primitivShape_t *shape;
  EXPECT_EQ(PRIMITIV_C_ERROR,
            ::primitivGetNodeShape(node, &shape));
  ::primitivDevice_t *device;
  EXPECT_EQ(PRIMITIV_C_ERROR,
            ::primitivGetDeviceFromNode(node, &device));
  float value;
  EXPECT_EQ(PRIMITIV_C_ERROR,
            ::primitivEvaluateNodeAsFloat(node, &value));
  std::size_t num_values = 20;
  float values[num_values];
  EXPECT_EQ(PRIMITIV_C_ERROR,
            ::primitivEvaluateNodeAsArray(node, values, &num_values));
  EXPECT_EQ(PRIMITIV_C_ERROR,
            ::primitivExecuteNodeBackward(node));
  ::primitivDeleteNode(node);
}

TEST_F(CGraphTest, CheckClear) {
  ::primitivResetStatus();
  ::primitivSetDefaultDevice(dev);

  ::primitivGraph_t *g;
  ASSERT_EQ(PRIMITIV_C_OK, ::primitivCreateGraph(&g));
  ::primitivSetDefaultGraph(g);

  uint32_t num;
  ::primitivGetGraphNumOperators(g, &num);
  EXPECT_EQ(0u, num);

  {
    ::primitivShape_t *shape;
    ASSERT_EQ(PRIMITIV_C_OK, ::primitivCreateShape(&shape));
    float values[] = {1};
    ::primitivNode_t *node1;
    ::primitivApplyNodeInput(shape, values, 1, nullptr, nullptr, &node1);
    ::primitivDeleteNode(node1);
    ::primitivNode_t *node2;
    ::primitivApplyNodeInput(shape, values, 1, nullptr, nullptr, &node2);
    ::primitivDeleteNode(node2);
    ::primitivGetGraphNumOperators(g, &num);
    EXPECT_EQ(2u, num);
  }

  ::primitivClearGraph(g);
  ::primitivGetGraphNumOperators(g, &num);
  EXPECT_EQ(0u, num);

  {
    ::primitivShape_t *shape;
    ASSERT_EQ(PRIMITIV_C_OK, ::primitivCreateShape(&shape));
    float values[] = {1};
    ::primitivNode_t *node;
    ::primitivApplyNodeInput(shape, values, 1, nullptr, nullptr, &node);
    ::primitivDeleteNode(node);
    ::primitivApplyNodeInput(shape, values, 1, nullptr, nullptr, &node);
    ::primitivDeleteNode(node);
    ::primitivApplyNodeInput(shape, values, 1, nullptr, nullptr, &node);
    ::primitivDeleteNode(node);
    ::primitivGetGraphNumOperators(g, &num);
    EXPECT_EQ(3u, num);
  }

  ::primitivClearGraph(g);
  ::primitivGetGraphNumOperators(g, &num);
  EXPECT_EQ(0u, num);

  // Clear empty graph.
  ::primitivClearGraph(g);
  ::primitivGetGraphNumOperators(g, &num);
  EXPECT_EQ(0u, num);

  ::primitivDeleteGraph(g);
}

TEST_F(CGraphTest, CheckForward) {
  ::primitivResetStatus();
  ::primitivSetDefaultDevice(dev);

  ::primitivGraph_t *g;
  ASSERT_EQ(PRIMITIV_C_OK, ::primitivCreateGraph(&g));
  ::primitivSetDefaultGraph(g);

  const float data1[] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  const float data3[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  vector<::primitivNode_t*> nodes;
  const uint32_t dims1[] = {2, 2};
  ::primitivShape_t *shape1;
  ASSERT_EQ(PRIMITIV_C_OK,
            ::primitivCreateShapeWithDims(dims1, 2, 3, &shape1));
  ::primitivNode_t *node1;
  ::primitivApplyNodeInput(shape1, data1, 12, nullptr, nullptr, &node1);
  nodes.emplace_back(node1);
  ::primitivShape_t *shape2;
  ASSERT_EQ(PRIMITIV_C_OK,
            ::primitivCreateShapeWithDims(dims1, 2, 1, &shape2));
  ::primitivNode_t *node2;
  ::primitivApplyNodeOnes(shape2, nullptr, nullptr, &node2);
  nodes.emplace_back(node2);
  ::primitivNode_t *node3;
  ::primitivApplyNodeInput(shape1, data3, 12, nullptr, nullptr, &node3);
  nodes.emplace_back(node3);
  ::primitivNode_t *node4;
  ::primitivApplyNodeAdd(nodes[0], nodes[1], &node4);
  nodes.emplace_back(node4);
  ::primitivNode_t *node5;
  ::primitivApplyNodeSubtract(nodes[1], nodes[2], &node5);
  nodes.emplace_back(node5);
  ::primitivNode_t *node6;
  ::primitivApplyNodeMultiply(nodes[3], nodes[4], &node6);
  nodes.emplace_back(node6);
  ::primitivNode_t *node7;
  ::primitivApplyNodeAddXC(nodes[5], 1, &node7);
  nodes.emplace_back(node7);
  ::primitivNode_t *node8;
  ::primitivApplyNodeSum(nodes[6], 0, &node8);
  nodes.emplace_back(node8);
  ::primitivNode_t *node9;
  ::primitivApplyNodeSum(nodes[7], 1, &node9);
  nodes.emplace_back(node9);
  ::primitivNode_t *node10;
  ::primitivApplyNodeBatchSum(nodes[8], &node10);
  nodes.emplace_back(node10);

  EXPECT_EQ(10u, nodes.size());
  uint32_t num;
  ::primitivGetGraphNumOperators(g, &num);
  EXPECT_EQ(10u, num);

  // Dump the graph to the output log.
  std::size_t length = 0u;
  ::primitivDumpGraph(g, "dot", nullptr, &length);
  EXPECT_GT(length, 0u);
  char str[length];
  ::primitivDumpGraph(g, "dot", str, &length);
  std::cout << str << std::endl;

  // Check all shapes and devices.
  const uint32_t dims2[] = {1, 2};
  ::primitivShape_t *shape3;
  ASSERT_EQ(PRIMITIV_C_OK,
            ::primitivCreateShapeWithDims(dims2, 2, 3, &shape3));
  const uint32_t dims3[] = {};
  ::primitivShape_t *shape4;
  ASSERT_EQ(PRIMITIV_C_OK,
            ::primitivCreateShapeWithDims(dims3, 0, 3, &shape4));
  ::primitivShape_t *shape5;
  ASSERT_EQ(PRIMITIV_C_OK,
            ::primitivCreateShapeWithDims(dims3, 0, 1, &shape5));
  const vector<::primitivShape_t*> expected_shapes {
    shape1, shape2, shape1,
    shape1, shape1, shape1,
    shape1,
    shape3, shape4, shape5,
  };
  for (std::uint32_t i = 0; i < nodes.size(); ++i) {
    ::primitivShape_t *shape;
    ::primitivDevice_t *device;
    ::primitivGetNodeShape(nodes[i], &shape);
    ::primitivGetDeviceFromNode(nodes[i], &device);
    PRIMITIV_C_BOOL eq;
    ::primitivIsShapeEqualTo(expected_shapes[i], shape, &eq);
    EXPECT_TRUE(eq);
    EXPECT_EQ(dev, device);
    ::primitivDeleteShape(shape);
    // ::primitiv_Device_delete(device);  // do not delete the reference
  }

  const ::primitivTensor_t *tensor;
  ::primitivExecuteGraphForward(g, nodes.back(), &tensor);
  // ::primitiv_Tensor_delete(const_cast<::primitiv_Tensor*>(tensor));
  // do not delete the reference

  // Check all node values.
  const vector<std::vector<float>> expected_values {
    {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4},
    {1, 1, 1, 1},
    {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2},
    {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5},
    {1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1},
    {2, 3, 4, 5, 0, 0, 0, 0, -2, -3, -4, -5},
    {3, 4, 5, 6, 1, 1, 1, 1, -1, -2, -3, -4},
    {7, 11, 2, 2, -3, -7},
    {18, 4, -10},
    {12},
  };
  for (std::uint32_t i = 0; i < nodes.size(); ++i) {
    // This forward method has no effect and only returns the reference to the
    // inner value.
    const ::primitivTensor_t *val;
    ::primitivExecuteGraphForward(g, nodes[i], &val);
    PRIMITIV_C_BOOL valid;
    ::primitivIsValidTensor(val, &valid);
    ASSERT_TRUE(valid);

    std::size_t size1;
    ::primitivEvaluateTensorAsArray(val, nullptr, &size1);
    float array1[size1];
    ::primitivEvaluateTensorAsArray(val, array1, &size1);
    float *expected_array = const_cast<float*>(&(expected_values[i])[0]);
    EXPECT_TRUE(array_match(expected_array, array1, size1));

    std::size_t size2;
    ::primitivEvaluateNodeAsArray(nodes[i], nullptr, &size2);
    float array2[size2];
    ::primitivEvaluateNodeAsArray(nodes[i], array2, &size2);
    EXPECT_TRUE(array_match(expected_array, array2, size2));
  }

  ::primitivDeleteShape(shape1);
  ::primitivDeleteShape(shape2);
  ::primitivDeleteShape(shape3);
  ::primitivDeleteShape(shape4);
  ::primitivDeleteShape(shape5);
  for (std::uint32_t i = 0; i < nodes.size(); ++i) {
    ::primitivDeleteNode(nodes[i]);
  }
  ::primitivDeleteGraph(g);
}

TEST_F(CGraphTest, CheckMultipleReturnValues) {
  ::primitivResetStatus();
  ::primitivSetDefaultDevice(dev);

  ::primitivGraph_t *g;
  ASSERT_EQ(PRIMITIV_C_OK, ::primitivCreateGraph(&g));
  ::primitivSetDefaultGraph(g);

  const uint32_t dims[] = {9};
  ::primitivShape_t *shape;
  ASSERT_EQ(PRIMITIV_C_OK,
            ::primitivCreateShapeWithDims(dims, 1, 1, &shape));
  const float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  ::primitivParameter_t *px;
  ASSERT_EQ(PRIMITIV_C_OK,
            ::primitivCreateParameterWithValues(shape, data, 9, nullptr, &px));
  ::primitivNode_t *x;
  ::primitivApplyNodeParameter(px, nullptr, &x);

  ::primitivNode_t *ys[3];
  ASSERT_EQ(PRIMITIV_C_OK, ::primitivApplyNodeSplit(x, 0, 3, ys));
  std::size_t length = 0u;
  ::primitivEvaluateNodeAsArray(ys[0], nullptr, &length);
  float y0_values[length];
  ::primitivEvaluateNodeAsArray(ys[0], y0_values, &length);
  float y0_expected[] = {1, 2, 3};
  EXPECT_TRUE(array_match(y0_expected, y0_values, length));
  ::primitivEvaluateNodeAsArray(ys[1], nullptr, &length);
  float y1_values[length];
  ::primitivEvaluateNodeAsArray(ys[1], y1_values, &length);
  float y1_expected[] = {4, 5, 6};
  EXPECT_TRUE(array_match(y1_expected, y1_values, length));
  ::primitivEvaluateNodeAsArray(ys[2], nullptr, &length);
  float y2_values[length];
  ::primitivEvaluateNodeAsArray(ys[2], y2_values, &length);
  float y2_expected[] = {7, 8, 9};
  EXPECT_TRUE(array_match(y2_expected, y2_values, length));

  ASSERT_EQ(PRIMITIV_C_OK, ::primitivResetParameterGradients(px));

  ASSERT_EQ(PRIMITIV_C_OK, ::primitivExecuteNodeBackward(ys[1]));
  const ::primitivTensor_t *grad0;
  ::primitivGetParameterGradient(px, &grad0);
  ::primitivEvaluateTensorAsArray(grad0, nullptr, &length);
  float grad0_values[length];
  ::primitivEvaluateTensorAsArray(grad0, grad0_values, &length);
  float grad0_expected[] {0, 0, 0, 1, 1, 1, 0, 0, 0};
  EXPECT_TRUE(array_match(grad0_expected, grad0_values, length));

  ::primitivNode_t *y3;
  ::primitivApplyNodeAdd(ys[0], ys[1], &y3);
  ASSERT_EQ(PRIMITIV_C_OK, ::primitivExecuteNodeBackward(y3));
  const ::primitivTensor_t *grad1;
  ::primitivGetParameterGradient(px, &grad1);
  ::primitivEvaluateTensorAsArray(grad1, nullptr, &length);
  float grad1_values[length];
  ::primitivEvaluateTensorAsArray(grad1, grad1_values, &length);
  float grad1_expected[] {1, 1, 1, 2, 2, 2, 0, 0, 0};
  EXPECT_TRUE(array_match(grad1_expected, grad1_values, length));

  ::primitivNode_t *y4;
  ::primitivApplyNodeSumNodes(ys, 3, &y4);
  ASSERT_EQ(PRIMITIV_C_OK, ::primitivExecuteNodeBackward(y4));
  const ::primitivTensor_t *grad2;
  ::primitivGetParameterGradient(px, &grad2);
  ::primitivEvaluateTensorAsArray(grad2, nullptr, &length);
  float grad2_values[length];
  ::primitivEvaluateTensorAsArray(grad2, grad2_values, &length);
  float grad2_expected[] {2, 2, 2, 3, 3, 3, 1, 1, 1};
  EXPECT_TRUE(array_match(grad2_expected, grad2_values, length));

  ::primitivNode_t *ys2[2];
  ASSERT_EQ(PRIMITIV_C_ERROR, ::primitivApplyNodeSplit(x, 0, 2, ys2));

  ::primitivDeleteShape(shape);
  ::primitivDeleteParameter(px);
  ::primitivDeleteNode(x);
  ::primitivDeleteNode(ys[0]);
  ::primitivDeleteNode(ys[1]);
  ::primitivDeleteNode(ys[2]);
  ::primitivDeleteNode(y3);
  ::primitivDeleteNode(y4);
  ::primitivDeleteGraph(g);
}

}  // namespace c
}  // namespace primitiv
