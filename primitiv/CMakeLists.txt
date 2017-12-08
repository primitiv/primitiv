# Build rules of the primitiv core library.

# Base libraries.
set(primitiv_base_HDRS
  arithmetic.h
  basic_functions.h
  composite_functions.h
  device.h
  error.h
  file_format.h
  functions.h
  graph.h
  initializer.h
  initializer_impl.h
  memory_pool.h
  mixins.h
  model.h
  naive_device.h
  numeric_utils.h
  operator.h
  operator_impl.h
  optimizer.h
  optimizer_impl.h
  parameter.h
  primitiv.h
  random.h
  shape.h
  shape_ops.h
  string_utils.h
  tensor.h
  type_traits.h
)
set(primitiv_base_SRCS
  device.cc
  graph.cc
  initializer_impl.cc
  memory_pool.cc
  model.cc
  naive_device.cc
  node_funcs.cc
  operator_impl.cc
  optimizer.cc
  optimizer_impl.cc
  parameter.cc
  shape.cc
  shape_ops.cc
  tensor.cc
  tensor_funcs.cc
)
install(FILES ${primitiv_base_HDRS} DESTINATION include/primitiv)

# MessagePack libraries.
set(primitiv_msgpack_HDRS
  msgpack/objects.h
  msgpack/reader.h
  msgpack/writer.h
)
install(FILES ${primitiv_msgpack_HDRS} DESTINATION include/primitiv/msgpack)

# Core library: all of above libraries.
set(primitiv_core_HDRS ${primitiv_base_HDRS} ${primitiv_msgpack_HDRS})
set(primitiv_core_SRCS ${primitiv_base_SRCS})

# Builds core library.
add_library(primitiv_core_OBJS OBJECT
  ${primitiv_core_HDRS}
  ${primitiv_core_SRCS})
set(primitiv_all_OBJS $<TARGET_OBJECTS:primitiv_core_OBJS>)
set(primitiv_all_DEPS)

# Build rules of the CUDA backend.
if(PRIMITIV_USE_CUDA)
  set(primitiv_cuda_HDRS
    primitiv_cuda.h
    cuda_device.h
    cuda_utils.h)
  set(primitiv_cuda_SRCS
    cuda_device.cu
    cuda_utils.cu)
  install(FILES ${primitiv_cuda_HDRS} DESTINATION include/primitiv)

  list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -O3 -Xcompiler -fPIC")

  # Workaround for some systems.
  #list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES")
  #list(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED")

  # Supported architectures.
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=sm_35")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_37,code=sm_37")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_52,code=sm_52")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_52,code=compute_52")

  cuda_compile(primitiv_cuda_OBJS
    ${primitiv_core_HDRS}
    ${primitiv_cuda_HDRS}
    ${primitiv_cuda_SRCS})
  list(APPEND primitiv_all_OBJS ${primitiv_cuda_OBJS})
  list(APPEND primitiv_all_DEPS
    ${CUDA_LIBRARIES}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_curand_LIBRARY})
endif()

# Build rules of the OpenCL backend.
if(PRIMITIV_USE_OPENCL)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/opencl_device_kernel.inc
    COMMAND
      ${CMAKE_COMMAND}
      -P
      ${PROJECT_SOURCE_DIR}/cmake/generate_char_array.cmake
      ${CMAKE_CURRENT_SOURCE_DIR}/opencl_device_kernel.cl
      ${CMAKE_CURRENT_BINARY_DIR}/opencl_device_kernel.inc
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/opencl_device_kernel.cl)
  add_custom_target(primitiv_opencl_KERNEL_DATA DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/opencl_device_kernel.inc)

  set(primitiv_opencl_HDRS
    primitiv_opencl.h
    opencl_device.h)
  set(primitiv_opencl_SRCS
    opencl_device.cc)
  install(FILES ${primitiv_opencl_HDRS} DESTINATION include/primitiv)

  add_library(primitiv_opencl_OBJS OBJECT
    ${primitiv_base_HDRS}
    ${primitiv_opencl_HDRS}
    ${primitiv_opencl_SRCS})
  add_dependencies(primitiv_opencl_OBJS primitiv_opencl_KERNEL_DATA)

  list(APPEND primitiv_all_OBJS $<TARGET_OBJECTS:primitiv_opencl_OBJS>)
  list(APPEND primitiv_all_DEPS
    ${OpenCL_LIBRARIES}
    ${CLBLAS_LIBRARIES})
endif()

# Builds the integrated binary.
if(PRIMITIV_BUILD_STATIC_LIBRARY)
  add_library(primitiv STATIC ${primitiv_all_OBJS})
else()
  add_library(primitiv SHARED ${primitiv_all_OBJS})
endif()
target_link_libraries(primitiv ${primitiv_all_DEPS})

install(TARGETS primitiv DESTINATION lib)
