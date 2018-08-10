#ifndef PRIMITIV_C_FUNCTIONS_H_
#define PRIMITIV_C_FUNCTIONS_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>
#include <primitiv/c/graph.h>
#include <primitiv/c/parameter.h>
#include <primitiv/c/shape.h>
#include <primitiv/c/tensor.h>

#define PRIMITIV_C_DECL_UNARY_FUNC(name) \
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNode##name( \
    const primitivNode_t *x, primitivNode_t **y); \
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensor##name( \
    const primitivTensor_t *x, primitivTensor_t **y);

#define PRIMITIV_C_DECL_BINARY_OP(name) \
PRIMITIV_C_API PRIMITIV_C_STATUS \
primitivApplyNode##name##XC( \
    const primitivNode_t *x, float k, primitivNode_t **y); \
PRIMITIV_C_API PRIMITIV_C_STATUS \
primitivApplyNode##name##CX( \
    float k, const primitivNode_t *x, primitivNode_t **y); \
PRIMITIV_C_API PRIMITIV_C_STATUS \
primitivApplyNode##name( \
    const primitivNode_t *a, const primitivNode_t *b, primitivNode_t **y); \
PRIMITIV_C_API PRIMITIV_C_STATUS \
primitivApplyTensor##name##XC( \
    const primitivTensor_t *x, float k, primitivTensor_t **y); \
PRIMITIV_C_API PRIMITIV_C_STATUS \
primitivApplyTensor##name##CX( \
    float k, const primitivTensor_t *x, primitivTensor_t **y); \
PRIMITIV_C_API PRIMITIV_C_STATUS \
primitivApplyTensor##name( \
    const primitivTensor_t *a, const primitivTensor_t *b, primitivTensor_t **y);

PRIMITIV_C_DECL_UNARY_FUNC(Positive);
PRIMITIV_C_DECL_UNARY_FUNC(Negative);
PRIMITIV_C_DECL_BINARY_OP(Add);
PRIMITIV_C_DECL_BINARY_OP(Subtract);
PRIMITIV_C_DECL_BINARY_OP(Multiply);
PRIMITIV_C_DECL_BINARY_OP(Divide);
PRIMITIV_C_DECL_BINARY_OP(Pow);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodePowN(
    const primitivNode_t *x, int32_t k, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorPowN(
    const primitivTensor_t *x, int32_t k, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeInput(
    const primitivShape_t *shape, const float *data, size_t n,
    primitivDevice_t *dev, primitivGraph_t *g, primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorInput(
    const primitivShape_t *shape, const float *data, size_t n,
    primitivDevice_t *dev, primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeParameter(
    primitivParameter_t *param, primitivGraph_t *g, primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorParameter(
    primitivParameter_t *param, primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeCopy(
    const primitivNode_t *x, primitivDevice_t *dev, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorCopy(
    const primitivTensor_t *x, primitivDevice_t *dev, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodePick(
    const primitivNode_t *x, const uint32_t *ids, size_t n, uint32_t dim,
    primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorPick(
    const primitivTensor_t *x, const uint32_t *ids, size_t n, uint32_t dim,
    primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeSlice(
    const primitivNode_t *x, uint32_t dim, uint32_t lower, uint32_t upper,
    primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorSlice(
    const primitivTensor_t *x, uint32_t dim, uint32_t lower, uint32_t upper,
    primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeSplit(
    const primitivNode_t *x, uint32_t dim, uint32_t n, primitivNode_t **ys);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorSplit(
    const primitivTensor_t *x, uint32_t dim, uint32_t n, primitivTensor_t **ys);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeConcat(
    const primitivNode_t *const *xs, size_t n, uint32_t dim,
    primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorConcat(
    const primitivTensor_t *const *xs, size_t n, uint32_t dim,
    primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeReshape(
    const primitivNode_t *x, const primitivShape_t *new_shape,
    primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorReshape(
    const primitivTensor_t *x, const primitivShape_t *new_shape,
    primitivTensor_t **y);

PRIMITIV_C_DECL_UNARY_FUNC(Flatten);
PRIMITIV_C_DECL_UNARY_FUNC(Transpose);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeFlip(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorFlip(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodePermuteDims(
    const primitivNode_t *x, const uint32_t *perm, size_t n,
    primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorPermuteDims(
    const primitivTensor_t *x, const uint32_t *perm, size_t n,
    primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeMatmul(
    const primitivNode_t *a, const primitivNode_t *b, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorMatmul(
    const primitivTensor_t *a, const primitivTensor_t *b, primitivTensor_t **y);

PRIMITIV_C_DECL_UNARY_FUNC(Abs);
PRIMITIV_C_DECL_UNARY_FUNC(Sqrt);
PRIMITIV_C_DECL_UNARY_FUNC(Exp);
PRIMITIV_C_DECL_UNARY_FUNC(Log);
PRIMITIV_C_DECL_UNARY_FUNC(Tanh);
PRIMITIV_C_DECL_UNARY_FUNC(Sigmoid);
PRIMITIV_C_DECL_UNARY_FUNC(Softplus);
PRIMITIV_C_DECL_UNARY_FUNC(Sin);
PRIMITIV_C_DECL_UNARY_FUNC(Cos);
PRIMITIV_C_DECL_UNARY_FUNC(Tan);
PRIMITIV_C_DECL_UNARY_FUNC(Relu);
PRIMITIV_C_DECL_UNARY_FUNC(Lrelu);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodePrelu(
    const primitivNode_t *x, float a, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorPrelu(
    const primitivTensor_t *x, float a, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeElu(
    const primitivNode_t *x, float a, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorElu(
    const primitivTensor_t *x, float a, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeMax(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorMax(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeMin(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorMin(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeSum(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorSum(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeBroadcast(
    const primitivNode_t *x, uint32_t dim, uint32_t size, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorBroadcast(
    const primitivTensor_t *x, uint32_t dim, uint32_t size,
    primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeLogsumexp(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorLogsumexp(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeLogSoftmax(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorLogSoftmax(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeSoftmax(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorSoftmax(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeSoftmaxCrossEntropy(
    const primitivNode_t *x, const primitivNode_t *t, uint32_t dim,
    primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorSoftmaxCrossEntropy(
    const primitivTensor_t *x, const primitivTensor_t *t, uint32_t dim,
    primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS
primitivApplyNodeSoftmaxCrossEntropyWithArray(
    const primitivNode_t *x, const uint32_t *ids, size_t n, uint32_t dim,
    primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS
primitivApplyTensorSoftmaxCrossEntropyWithArray(
    const primitivTensor_t *x, const uint32_t *ids, size_t n, uint32_t dim,
    primitivTensor_t **y);

PRIMITIV_C_DECL_UNARY_FUNC(StopGradient);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeConv2d(
    const primitivNode_t *x, const primitivNode_t *w,
    uint32_t padding0, uint32_t padding1,
    uint32_t stride0, uint32_t stride1,
    uint32_t dilation0, uint32_t dilation1,
    primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorConv2d(
    const primitivTensor_t *x, const primitivTensor_t *w,
    uint32_t padding0, uint32_t padding1,
    uint32_t stride0, uint32_t stride1,
    uint32_t dilation0, uint32_t dilation1,
    primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeMaxPool2d(
    const primitivNode_t *x,
    uint32_t window0, uint32_t window1,
    uint32_t padding0, uint32_t padding1,
    uint32_t stride0, uint32_t stride1,
    primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorMaxPool2d(
    const primitivTensor_t *x,
    uint32_t window0, uint32_t window1,
    uint32_t padding0, uint32_t padding1,
    uint32_t stride0, uint32_t stride1,
    primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeBatchPick(
    const primitivNode_t *x, const uint32_t *ids, size_t n,
    primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorBatchPick(
    const primitivTensor_t *x, const uint32_t *ids, size_t n,
    primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeBatchSlice(
    const primitivNode_t *x, uint32_t lower, uint32_t upper,
    primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorBatchSlice(
    const primitivTensor_t *x, uint32_t lower, uint32_t upper,
    primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeBatchSplit(
    const primitivNode_t *x, uint32_t n, primitivNode_t **ys);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorBatchSplit(
    const primitivTensor_t *x, uint32_t n, primitivTensor_t **ys);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeBatchConcat(
    const primitivNode_t *const *xs, size_t n, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorBatchConcat(
    const primitivTensor_t *const *xs, size_t n, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeBatchSum(
    const primitivNode_t *x, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorBatchSum(
    const primitivTensor_t *x, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeConstant(
    const primitivShape_t *shape, float k, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorConstant(
    const primitivShape_t *shape, float k, primitivDevice_t *dev,
    primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeIdentity(
    uint32_t size, primitivDevice_t *dev, primitivGraph_t *g,
    primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorIdentity(
    uint32_t size, primitivDevice_t *dev, primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeRandomBernoulli(
    const primitivShape_t *shape, float p, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorRandomBernoulli(
    const primitivShape_t *shape, float p, primitivDevice_t *dev,
    primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeRandomUniform(
    const primitivShape_t *shape, float lower, float upper,
    primitivDevice_t *dev, primitivGraph_t *g, primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorRandomUniform(
    const primitivShape_t *shape, float lower, float upper,
    primitivDevice_t *dev, primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeRandomNormal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorRandomNormal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeRandomLogNormal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorRandomLogNormal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeRandomGumbel(
    const primitivShape_t *shape, float mu, float beta, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorRandomGumbel(
    const primitivShape_t *shape, float mu, float beta, primitivDevice_t *dev,
    primitivTensor_t **newobj);

PRIMITIV_C_DECL_UNARY_FUNC(Selu);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeSumNodes(
    const primitivNode_t *const *xs, size_t n, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorSumTensors(
    const primitivTensor_t *const *xs, size_t n, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeMean(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorMean(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeMeanNodes(
    const primitivNode_t *const *xs, size_t n, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorMeanTensors(
    const primitivTensor_t *const *xs, size_t n, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeBatchMean(
    const primitivNode_t *x, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorBatchMean(
    const primitivTensor_t *x, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeBatchNormalize(
    const primitivNode_t *x, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorBatchNormalize(
    const primitivTensor_t *x, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeZeros(
    const primitivShape_t *shape, primitivDevice_t *dev, primitivGraph_t *g,
    primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorZeros(
    const primitivShape_t *shape, primitivDevice_t *dev,
    primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeOnes(
    const primitivShape_t *shape, primitivDevice_t *dev, primitivGraph_t *g,
    primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorOnes(
    const primitivShape_t *shape, primitivDevice_t *dev,
    primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyNodeDropout(
    const primitivNode_t *x, float rate, PRIMITIV_C_BOOL enabled,
    primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivApplyTensorDropout(
    const primitivTensor_t *x, float rate, PRIMITIV_C_BOOL enabled,
    primitivTensor_t **y);

#undef PRIMITIV_C_DECL_UNARY_FUNC
#undef PRIMITIV_C_DECL_BINARY_OP

#endif  // PRIMITIV_C_FUNCTIONS_H_
