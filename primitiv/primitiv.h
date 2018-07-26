#ifndef PRIMITIV_PRIMITIV_H_
#define PRIMITIV_PRIMITIV_H_

#include <primitiv/config.h>

// This header file describes some include directives and may help users to use
// the primitiv library.
#include <primitiv/core/error.h>
#include <primitiv/core/functions.h>
#include <primitiv/core/graph.h>
#include <primitiv/core/initializer_impl.h>
#include <primitiv/core/model.h>
#include <primitiv/core/naive_device.h>
#include <primitiv/core/parameter.h>
#include <primitiv/core/shape.h>
#include <primitiv/core/tensor.h>
#include <primitiv/core/optimizer_impl.h>

// Header files for specific device classes.
#ifdef PRIMITIV_USE_EIGEN
#include <primitiv/core/eigen_device.h>
#endif  // PRIMITIV_USE_EIGEN
#ifdef PRIMITIV_USE_CUDA
#include <primitiv/core/cuda_device.h>
#include <primitiv/core/cuda16_device.h>
#endif  // PRIMITIV_USE_CUDA
#ifdef PRIMITIV_USE_OPENCL
#include <primitiv/core/opencl_device.h>
#endif  // PRIMITIV_USE_OPENCL

#endif  // PRIMITIV_PRIMITIV_H_
