# Build rules of the primitiv core library.

# Global configuration header
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
)

# Version header.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/version.h
  @ONLY
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
  ${CMAKE_CURRENT_BINARY_DIR}/version.h
  DESTINATION include/primitiv
)

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

# Build rules of the Eigen backend.
if(PRIMITIV_USE_EIGEN)
  set(primitiv_eigen_HDRS eigen_device.h)
  set(primitiv_eigen_SRCS eigen_device.cc)
  install(FILES ${primitiv_eigen_HDRS} DESTINATION include/primitiv)

  add_library(primitiv_eigen_OBJS OBJECT
    ${primitiv_base_HDRS}
    ${primitiv_eigen_HDRS}
    ${primitiv_eigen_SRCS})

  list(APPEND primitiv_all_OBJS $<TARGET_OBJECTS:primitiv_eigen_OBJS>)
endif()

# Build rules of the CUDA backend.
if(PRIMITIV_USE_CUDA)
  set(primitiv_cuda_HDRS cuda_device.h cuda_utils.h)
  set(primitiv_cuda_SRCS cuda_device.cu cuda_utils.cu)
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

  set(primitiv_opencl_HDRS opencl_device.h)
  set(primitiv_opencl_SRCS opencl_device.cc)
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

# Builds C API
if(PRIMITIV_BUILD_C_API)
  set(primitiv_c_HDRS
    c/api.h
    c/define.h
    c/device.h
    c/functions.h
    c/graph.h
    c/initializer.h
    c/initializer_impl.h
    c/internal.h
    c/model.h
    c/naive_device.h
    c/optimizer.h
    c/optimizer_impl.h
    c/parameter.h
    c/shape.h
    c/status.h
    c/tensor.h
    c/utils.h
  )
  set(primitiv_c_SRCS
    c/device.cc
    c/functions.cc
    c/graph.cc
    c/initializer.cc
    c/initializer_impl.cc
    c/model.cc
    c/naive_device.cc
    c/optimizer.cc
    c/optimizer_impl.cc
    c/parameter.cc
    c/shape.cc
    c/status.cc
    c/tensor.cc
    c/utils.cc
  )
  install(FILES ${primitiv_c_HDRS} DESTINATION include/primitiv/c)

  add_library(primitiv_c_OBJS OBJECT
      ${primitiv_c_HDRS}
      ${primitiv_c_SRCS})
  set(primitiv_c_all_OBJS $<TARGET_OBJECTS:primitiv_c_OBJS>)

  if(PRIMITIV_USE_CUDA)
    set(primitiv_c_cuda_HDRS
      c/api_cuda.h
      c/cuda_device.h)
    set(primitiv_c_cuda_SRCS
      c/cuda_device.cc)
    install(FILES ${primitiv_c_cuda_HDRS} DESTINATION include/primitiv/c)

    add_library(primitiv_c_cuda_OBJS OBJECT
      ${primitiv_c_cuda_HDRS}
      ${primitiv_c_cuda_SRCS})
    list(APPEND primitiv_c_all_OBJS $<TARGET_OBJECTS:primitiv_c_cuda_OBJS>)
  endif()

  if(PRIMITIV_USE_OPENCL)
    set(primitiv_c_opencl_HDRS
      c/api_opencl.h
      c/opencl_device.h)
    set(primitiv_c_opencl_SRCS
      c/opencl_device.cc)
    install(FILES ${primitiv_c_opencl_HDRS} DESTINATION include/primitiv/c)

    add_library(primitiv_c_opencl_OBJS OBJECT
      ${primitiv_c_opencl_HDRS}
      ${primitiv_c_opencl_SRCS})
    list(APPEND primitiv_c_all_OBJS $<TARGET_OBJECTS:primitiv_c_opencl_OBJS>)
  endif()

  if(PRIMITIV_BUILD_STATIC_LIBRARY)
    add_library(primitiv_c STATIC ${primitiv_c_all_OBJS})
  else()
    add_library(primitiv_c SHARED ${primitiv_c_all_OBJS})
  endif()
  target_link_libraries(primitiv_c primitiv)

  install(TARGETS primitiv_c DESTINATION lib)
endif()
