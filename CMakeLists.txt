cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

project(primitiv_c)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(PRIMITIV_BUILD_STATIC_LIBRARY "Builds static library." OFF)
option(PRIMITIV_USE_CUDA "Enables the CUDA backend." OFF)
option(PRIMITIV_USE_OPENCL "Enables the OpenCL backend." OFF)

# C++ version
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# compiler settings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Werror -fPIC")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Werror -fPIC")
else()
  message(WARNING "primitiv may not support the detected compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

find_package(Primitiv REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_BINARY_DIR}
  ${PRIMITIV_INCLUDE_DIR}
)

link_directories(
  ${PRIMITIV_LIBRARY_DIR}
)

add_subdirectory(primitiv_c)
