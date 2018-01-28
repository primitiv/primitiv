#ifndef PRIMITIV_C_FUNCTIONS_H_
#define PRIMITIV_C_FUNCTIONS_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>
#include <primitiv/c/graph.h>
#include <primitiv/c/parameter.h>
#include <primitiv/c/shape.h>
#include <primitiv/c/tensor.h>

#define PRIMITIV_C_DECL_UNARY_FUNC(name) \
PRIMITIV_C_API PRIMITIV_C_STATUS primitivNode##name( \
    const primitivNode_t *x, primitivNode_t **y); \
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensor##name( \
    const primitivTensor_t *x, primitivTensor_t **y);

#define PRIMITIV_C_DECL_BINARY_OP(name) \
PRIMITIV_C_API PRIMITIV_C_STATUS \
primitivNode##name##NodeConst( \
    const primitivNode_t *x, float k, primitivNode_t **y); \
PRIMITIV_C_API PRIMITIV_C_STATUS \
primitivNode##name##ConstNode( \
    float k, const primitivNode_t *x, primitivNode_t **y); \
PRIMITIV_C_API PRIMITIV_C_STATUS \
primitivNode##name##NodeNode( \
    const primitivNode_t *a, const primitivNode_t *b, primitivNode_t **y); \
PRIMITIV_C_API PRIMITIV_C_STATUS \
primitivTensor##name##TensorConst( \
    const primitivTensor_t *x, float k, primitivTensor_t **y); \
PRIMITIV_C_API PRIMITIV_C_STATUS \
primitivTensor##name##ConstTensor( \
    float k, const primitivTensor_t *x, primitivTensor_t **y); \
PRIMITIV_C_API PRIMITIV_C_STATUS \
primitivTensor##name##TensorTensor( \
    const primitivTensor_t *a, const primitivTensor_t *b, primitivTensor_t **y);

PRIMITIV_C_DECL_UNARY_FUNC(Positive);
PRIMITIV_C_DECL_UNARY_FUNC(Negative);
PRIMITIV_C_DECL_BINARY_OP(Add);
PRIMITIV_C_DECL_BINARY_OP(Subtract);
PRIMITIV_C_DECL_BINARY_OP(Multiply);
PRIMITIV_C_DECL_BINARY_OP(Divide);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeInput(
    const primitivShape_t *shape, const float *data, size_t n,
    primitivDevice_t *dev, primitivGraph_t *g, primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorInput(
    const primitivShape_t *shape, const float *data, size_t n,
    primitivDevice_t *dev, primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeParameter(
    primitivParameter_t *param, primitivGraph_t *g, primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorParameter(
    primitivParameter_t *param, primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeCopy(
    const primitivNode_t *x, primitivDevice_t *dev, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorCopy(
    const primitivTensor_t *x, primitivDevice_t *dev, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodePick(
    const primitivNode_t *x, const uint32_t *ids, size_t n, uint32_t dim,
    primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorPick(
    const primitivTensor_t *x, const uint32_t *ids, size_t n, uint32_t dim,
    primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeSlice(
    const primitivNode_t *x, uint32_t dim, uint32_t lower, uint32_t upper,
    primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorSlice(
    const primitivTensor_t *x, uint32_t dim, uint32_t lower, uint32_t upper,
    primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeConcat(
    const primitivNode_t *const *xs, size_t n, uint32_t dim, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorConcat(
    const primitivTensor_t *const *xs, size_t n, uint32_t dim,
    primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeReshape(
    const primitivNode_t *x, const primitivShape_t *new_shape, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorReshape(
    const primitivTensor_t *x, const primitivShape_t *new_shape,
    primitivTensor_t **y);

PRIMITIV_C_DECL_UNARY_FUNC(Flatten);
PRIMITIV_C_DECL_UNARY_FUNC(Transpose);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeMatmul(
    const primitivNode_t *a, const primitivNode_t *b, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorMatmul(
    const primitivTensor_t *a, const primitivTensor_t *b, primitivTensor_t **y);

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

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodePrelu(
    const primitivNode_t *x, float a, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorPrelu(
    const primitivTensor_t *x, float a, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeElu(
    const primitivNode_t *x, float a, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorElu(
    const primitivTensor_t *x, float a, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeSum(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorSum(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeBroadcast(
    const primitivNode_t *x, uint32_t dim, uint32_t size, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorBroadcast(
    const primitivTensor_t *x, uint32_t dim, uint32_t size, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeLogsumexp(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorLogsumexp(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeLogSoftmax(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorLogSoftmax(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeSoftmax(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorSoftmax(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeSoftmaxCrossEntropy(
    const primitivNode_t *x, const primitivNode_t *t, uint32_t dim,
    primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorSoftmaxCrossEntropy(
    const primitivTensor_t *x, const primitivTensor_t *t, uint32_t dim,
    primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeSoftmaxCrossEntropyWithArray(
    const primitivNode_t *x, const uint32_t *ids, size_t n, uint32_t dim,
    primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorSoftmaxCrossEntropyWithArray(
    const primitivTensor_t *x, const uint32_t *ids, size_t n, uint32_t dim,
    primitivTensor_t **y);

PRIMITIV_C_DECL_UNARY_FUNC(StopGradient);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeBatchSum(
    const primitivNode_t *x, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorBatchSum(
    const primitivTensor_t *x, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeConstant(
    const primitivShape_t *shape, float k, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorConstant(
    const primitivShape_t *shape, float k, primitivDevice_t *dev,
    primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeIdentity(
    uint32_t size, primitivDevice_t *dev, primitivGraph_t *g,
    primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorIdentity(
    uint32_t size, primitivDevice_t *dev, primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeRandomBernoulli(
    const primitivShape_t *shape, float p, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorRandomBernoulli(
    const primitivShape_t *shape, float p, primitivDevice_t *dev,
    primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeRandomUniform(
    const primitivShape_t *shape, float lower, float upper, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorRandomUniform(
    const primitivShape_t *shape, float lower, float upper, primitivDevice_t *dev,
    primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeRandomNormal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorRandomNormal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeRandomLogNormal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorRandomLogNormal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeRandomGumbel(
    const primitivShape_t *shape, float mu, float beta, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorRandomGumbel(
    const primitivShape_t *shape, float mu, float beta, primitivDevice_t *dev,
    primitivTensor_t **newobj);

PRIMITIV_C_DECL_BINARY_OP(Pow);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodePown(
    const primitivNode_t *x, uint32_t k, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorPown(
    const primitivTensor_t *x, uint32_t k, primitivTensor_t **y);

PRIMITIV_C_DECL_UNARY_FUNC(Selu);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeSumNodes(
    const primitivNode_t *const *xs, size_t n, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorSumTensors(
    const primitivTensor_t *const *xs, size_t n, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeMean(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorMean(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeMeanNodes(
    const primitivNode_t *const *xs, size_t n, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorMeanTensors(
    const primitivTensor_t *const *xs, size_t n, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeBatchMean(
    const primitivNode_t *x, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorBatchMean(
    const primitivTensor_t *x, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeBatchNormalize(
    const primitivNode_t *x, primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorBatchNormalize(
    const primitivTensor_t *x, primitivTensor_t **y);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeZeros(
    const primitivShape_t *shape, primitivDevice_t *dev, primitivGraph_t *g,
    primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorZeros(
    const primitivShape_t *shape, primitivDevice_t *dev,
    primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeOnes(
    const primitivShape_t *shape, primitivDevice_t *dev, primitivGraph_t *g,
    primitivNode_t **newobj);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorOnes(
    const primitivShape_t *shape, primitivDevice_t *dev,
    primitivTensor_t **newobj);

PRIMITIV_C_API PRIMITIV_C_STATUS primitivNodeDropout(
    const primitivNode_t *x, float rate, PRIMITIV_C_BOOL enabled,
    primitivNode_t **y);
PRIMITIV_C_API PRIMITIV_C_STATUS primitivTensorDropout(
    const primitivTensor_t *x, float rate, PRIMITIV_C_BOOL enabled,
    primitivTensor_t **y);

#undef PRIMITIV_C_DECL_UNARY_FUNC
#undef PRIMITIV_C_DECL_BINARY_OP

#endif  // PRIMITIV_C_FUNCTIONS_H_
