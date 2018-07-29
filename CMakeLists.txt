cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)

project(primitiv VERSION 0.4.0 LANGUAGES CXX)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(PRIMITIV_BUILD_C_API "Builds C API corresponding to the core library." OFF)
option(PRIMITIV_BUILD_STATIC_LIBRARY "Builds static library." OFF)
option(PRIMITIV_BUILD_TESTS "Builds test binaries." OFF)
option(PRIMITIV_BUILD_TESTS_PROBABILISTIC "Builds test cases that probabilistically fails." OFF)
option(PRIMITIV_GTEST_SOURCE_DIR "Source directory of Google Test library." "")
option(PRIMITIV_USE_CACHE "Enables cached values in some functions but needs more memory." OFF)
option(PRIMITIV_USE_EIGEN "Enables the Eigen backend." OFF)
option(PRIMITIV_USE_CUDA "Enables the CUDA backend." OFF)
option(PRIMITIV_USE_CUDNN "Enables the cuDNN library (PRIMITIV_USE_CUDA is also necessary)." OFF)
option(PRIMITIV_USE_OPENCL "Enables the OpenCL backend." OFF)

# C++ version
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# compiler settings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -Werror -fPIC ${CMAKE_CXX_FLAGS}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -Werror -fPIC ${CMAKE_CXX_FLAGS}")
else()
  message(WARNING "primitiv may not support the detected compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

# External packages.
if(PRIMITIV_USE_EIGEN)
  find_package(Eigen3 3.3.0 REQUIRED)
endif()
if(PRIMITIV_USE_CUDA)
  find_package(CUDA 8.0 REQUIRED)
  if(PRIMITIV_USE_CUDNN)
    find_package(CuDNN 5.1.0 REQUIRED)
  endif()
endif()
if(PRIMITIV_USE_OPENCL)
  find_package(OpenCL REQUIRED)
  find_package(CLHPP REQUIRED)
  find_package(CLBlast 1.2.0 REQUIRED)
endif()

# Include directories.
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_BINARY_DIR})
if(PRIMITIV_USE_EIGEN)
  include_directories(${EIGEN3_INCLUDE_DIR})
endif()
if(PRIMITIV_USE_CUDA)
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
endif()
if(PRIMITIV_USE_OPENCL)
  include_directories(${CLHPP_INCLUDE_DIR})
  include_directories(${CLBLAST_INCLUDE_DIR})
endif()

# core library
add_subdirectory(primitiv)

# tests
if(PRIMITIV_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
