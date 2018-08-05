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
#include <primitiv/core/parameter.h>
#include <primitiv/core/shape.h>
#include <primitiv/core/tensor.h>
#include <primitiv/core/optimizer_impl.h>
#include <primitiv/devices/naive/device.h>

// Header files for specific device classes.
#ifdef PRIMITIV_USE_EIGEN
#include <primitiv/devices/eigen/device.h>
#endif  // PRIMITIV_USE_EIGEN
#ifdef PRIMITIV_USE_CUDA
#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda16/device.h>
#endif  // PRIMITIV_USE_CUDA
#ifdef PRIMITIV_USE_OPENCL
#include <primitiv/devices/opencl/device.h>
#endif  // PRIMITIV_USE_OPENCL

#endif  // PRIMITIV_PRIMITIV_H_
