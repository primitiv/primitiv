#ifndef PRIMITIV_C_API_H_
#define PRIMITIV_C_API_H_

#include <primitiv/config.h>

#include <primitiv/c/graph.h>
#include <primitiv/c/initializer_impl.h>
#include <primitiv/c/model.h>
#include <primitiv/c/functions.h>
#include <primitiv/c/parameter.h>
#include <primitiv/c/shape.h>
#include <primitiv/c/status.h>
#include <primitiv/c/tensor.h>
#include <primitiv/c/optimizer_impl.h>
#include <primitiv/c/devices/naive/device.h>

// Header files for specific device classes.
#ifdef PRIMITIV_USE_EIGEN
#include <primitiv/c/devices/eigen/device.h>
#endif  // PRIMITIV_USE_EIGEN
#ifdef PRIMITIV_USE_CUDA
#include <primitiv/c/devices/cuda/device.h>
#endif  // PRIMITIV_USE_CUDA
#ifdef PRIMITIV_USE_OPENCL
#include <primitiv/c/devices/opencl/device.h>
#endif  // PRIMITIV_USE_OPENCL

#endif  // PRIMITIV_C_API_H_
