#ifndef PRIMITIV_PRIMITIV_H_
#define PRIMITIV_PRIMITIV_H_

#include <primitiv/config.h>

// This header file describes some include directives and may help users to use
// the primitiv library.
#include <primitiv/error.h>
#include <primitiv/functions.h>
#include <primitiv/graph.h>
#include <primitiv/initializer_impl.h>
#include <primitiv/model.h>
#include <primitiv/naive_device.h>
#include <primitiv/parameter.h>
#include <primitiv/shape.h>
#include <primitiv/tensor.h>
#include <primitiv/optimizer_impl.h>

// Header files for specific device classes.
#ifdef PRIMITIV_USE_EIGEN
#include <primitiv/eigen_device.h>
#endif  // PRIMITIV_USE_EIGEN
#ifdef PRIMITIV_USE_CUDA
#include <primitiv/cuda_device.h>
#endif  // PRIMITIV_USE_CUDA
#ifdef PRIMITIV_USE_OPENCL
#include <primitiv/opencl_device.h>
#endif  // PRIMITIV_USE_OPENCL

#endif  // PRIMITIV_PRIMITIV_H_
