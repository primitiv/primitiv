cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)

project(primitiv VERSION 0.1.0 LANGUAGES CXX)

option(PRIMITIV_BUILD_STATIC_LIBRARY "Builds static library." OFF)
option(PRIMITIV_BUILD_TESTS "Builds test binaries." ON)
option(PRIMITIV_USE_CACHE "Enables cached values in some functions but needs more memory." OFF)
option(PRIMITIV_USE_CUDA "Finds CUDA library ant use it." OFF)

# compiler requirements and settings
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -Wall -Werror -fPIC")
endif()

# subproject settings
set(YAML_CPP_BUILD_TOOLS CACHE BOOL OFF)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
)

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
link_directories()

if(PRIMITIV_USE_CUDA)
  find_package(CUDA REQUIRED)
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  link_directories(${CUDA_LIBRARIES})
endif()

# sources
add_subdirectory(submodules/yaml-cpp)
add_subdirectory(primitiv)
if(PRIMITIV_BUILD_TESTS)
  enable_testing()
  add_subdirectory(submodules/googletest)
  add_subdirectory(test)
endif()
