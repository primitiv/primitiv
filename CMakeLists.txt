cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(primitiv VERSION 0.1.0 LANGUAGES CXX)

# compiler requirements and settings
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -Wall -Werror")
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
)

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
link_directories()

if(USE_CUDA)
  find_package(CUDA REQUIRED)
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  link_directories(${CUDA_LIBRARIES})
endif()

enable_testing()

# sources
add_subdirectory(primitiv)

# tests
add_subdirectory(submodules/googletest)
add_subdirectory(test)
