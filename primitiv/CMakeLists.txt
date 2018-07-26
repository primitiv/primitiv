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

# Combined header (won't be used by the library itself)
install(FILES primitiv.h DESTINATION include/primitiv)

# Core libraries.
set(primitiv_core_HDRS
  core/arithmetic.h
  core/basic_functions.h
  core/device.h
  core/error.h
  core/file_format.h
  core/functions.h
  core/graph.h
  core/initializer.h
  core/initializer_impl.h
  core/memory_pool.h
  core/mixins.h
  core/model.h
  core/naive_device.h
  core/numeric_utils.h
  core/operator.h
  core/operator_impl.h
  core/optimizer.h
  core/optimizer_impl.h
  core/parameter.h
  core/random.h
  core/shape.h
  core/shape_ops.h
  core/string_utils.h
  core/tensor.h
  core/type_traits.h
)
set(primitiv_core_SRCS
  core/device.cc
  core/graph.cc
  core/initializer_impl.cc
  core/memory_pool.cc
  core/model.cc
  core/node_funcs.cc
  core/operator_impl.cc
  core/optimizer.cc
  core/optimizer_impl.cc
  core/parameter.cc
  core/shape.cc
  core/shape_ops.cc
  core/tensor.cc
  core/tensor_funcs.cc
)
install(FILES ${primitiv_core_HDRS} DESTINATION include/primitiv/core)

# Code for the Naive device (internal; won't be installed)
file(GLOB primitiv_naive_devops_HDRS "device_ops/naive/*.h")
file(GLOB primitiv_naive_devops_SRCS "device_ops/naive/*.cc")

# MessagePack libraries.
set(primitiv_msgpack_HDRS
  msgpack/objects.h
  msgpack/reader.h
  msgpack/writer.h
)
install(FILES ${primitiv_msgpack_HDRS} DESTINATION include/primitiv/msgpack)

# Contrib libraries.
set(primitiv_contrib_HDRS
  contrib/functions.h
)
install(FILES ${primitiv_contrib_HDRS} DESTINATION include/primitiv/contrib)

# Minimal library: all of above.
set(primitiv_minimal_HDRS
  ${primitiv_core_HDRS}
  ${primitiv_msgpack_HDRS}
  ${primitiv_contrib_HDRS}
  ${primitiv_naive_devops_HDRS}
)
set(primitiv_minimal_SRCS
  ${primitiv_core_SRCS}
  ${primitiv_naive_devops_SRCS}
)

# Builds the minimal library.
add_library(primitiv_core_OBJS OBJECT
  ${primitiv_minimal_HDRS}
  ${primitiv_minimal_SRCS}
)
set(primitiv_all_OBJS $<TARGET_OBJECTS:primitiv_core_OBJS>)
set(primitiv_all_DEPS)

# Build rules of the Eigen backend.
if(PRIMITIV_USE_EIGEN)
  set(primitiv_eigen_HDRS core/eigen_device.h)
  file(GLOB primitiv_eigen_devops_HDRS "device_ops/eigen/*.h")
  file(GLOB primitiv_eigen_devops_SRCS "device_ops/eigen/*.cc")
  install(FILES ${primitiv_eigen_HDRS} DESTINATION include/primitiv/core)

  set(primitiv_eigen_COMPILE_FLAGS "")

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(primitiv_eigen_COMPILE_FLAGS
      "${primitiv_eigen_COMPILE_FLAGS} -march=native")
    # NOTE(odashi):
    # Explicitly disabling `-Wint-in-bool-context` to avoid warnings caused by
    # Eigen v3.3.4 or earlier.
    if(CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 7.0 OR
        CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
      if (EIGEN3_VERSION VERSION_LESS 3.3.5)
        set(primitiv_eigen_COMPILE_FLAGS
          "${primitiv_eigen_COMPILE_FLAGS} -Wno-int-in-bool-context")
      endif()
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(primitiv_eigen_COMPILE_FLAGS
      "${primitiv_eigen_COMPILE_FLAGS} -march=native")
  endif()

  set_source_files_properties(
    ${primitiv_eigen_devops_SRCS}
    PROPERTIES COMPILE_FLAGS ${primitiv_eigen_COMPILE_FLAGS}
  )

  add_library(primitiv_eigen_OBJS OBJECT
    ${primitiv_core_HDRS}
    ${primitiv_eigen_HDRS}
    ${primitiv_eigen_devops_HDRS}
    ${primitiv_eigen_devops_SRCS}
  )

  list(APPEND primitiv_all_OBJS $<TARGET_OBJECTS:primitiv_eigen_OBJS>)
endif()

# Build rules of the CUDA backend.
if(PRIMITIV_USE_CUDA)
  set(primitiv_cuda_HDRS core/cuda_device.h core/cuda16_device.h)
  set(primitiv_cuda_SRCS core/cuda_device.cu core/cuda16_device.cu)
  set(primitiv_cuda_internal_HDRS internal/cuda_utils.h)
  set(primitiv_cuda_internal_SRCS internal/cuda_utils.cu)
  file(GLOB primitiv_cuda_devops_HDRS "device_ops/cuda/*.h")
  file(GLOB primitiv_cuda_devops_SRCS "device_ops/cuda/*.cu")
  file(GLOB primitiv_cuda16_devops_HDRS "device_ops/cuda16/*.h")
  file(GLOB primitiv_cuda16_devops_SRCS "device_ops/cuda16/*.cu")
  install(FILES ${primitiv_cuda_HDRS} DESTINATION include/primitiv/core)

  list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -O3 -Xcompiler -fPIC")

  # Workaround for some systems.
  #list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES")
  #list(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED")

  #
  # Supported architectures.
  #

  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=sm_35")

  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_52,code=sm_52")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_52,code=compute_52")

  if(CUDA_VERSION VERSION_EQUAL 8.0 OR CUDA_VERSION VERSION_GREATER 8.0)
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_60,code=sm_60")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=sm_61")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=compute_61")
  endif()

  if(CUDA_VERSION VERSION_EQUAL 9.0 OR CUDA_VERSION VERSION_GREATER 9.0)
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_70,code=sm_70")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_70,code=compute_70")
  endif()

  cuda_compile(primitiv_cuda_OBJS
    ${primitiv_core_HDRS}
    ${primitiv_cuda_HDRS}
    ${primitiv_cuda_SRCS}
    ${primitiv_cuda_internal_HDRS}
    ${primitiv_cuda_internal_SRCS}
    ${primitiv_cuda_devops_HDRS}
    ${primitiv_cuda_devops_SRCS}
    ${primitiv_cuda16_devops_HDRS}
    ${primitiv_cuda16_devops_SRCS}
  )
  list(APPEND primitiv_all_OBJS ${primitiv_cuda_OBJS})
  list(APPEND primitiv_all_DEPS
    ${CUDA_LIBRARIES}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_curand_LIBRARY}
    ${CUDNN_LIBRARIES}
  )
endif()

# Build rules of the OpenCL backend.
if(PRIMITIV_USE_OPENCL)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/core/opencl_device_kernel.inc
    COMMAND
      ${CMAKE_COMMAND}
      -P
      ${PROJECT_SOURCE_DIR}/cmake/generate_char_array.cmake
      ${CMAKE_CURRENT_SOURCE_DIR}/core/opencl_device_kernel.cl
      ${CMAKE_CURRENT_BINARY_DIR}/core/opencl_device_kernel.inc
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/core/opencl_device_kernel.cl
  )
  add_custom_target(primitiv_opencl_KERNEL_DATA DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/core/opencl_device_kernel.inc
  )

  set(primitiv_opencl_HDRS core/opencl_device.h)
  set(primitiv_opencl_SRCS core/opencl_device.cc)
  install(FILES ${primitiv_opencl_HDRS} DESTINATION include/primitiv/core)

  add_library(primitiv_opencl_OBJS OBJECT
    ${primitiv_core_HDRS}
    ${primitiv_opencl_HDRS}
    ${primitiv_opencl_SRCS}
  )
  add_dependencies(primitiv_opencl_OBJS primitiv_opencl_KERNEL_DATA)

  list(APPEND primitiv_all_OBJS $<TARGET_OBJECTS:primitiv_opencl_OBJS>)
  list(APPEND primitiv_all_DEPS
    ${OpenCL_LIBRARIES}
    ${CLBLAST_LIBRARIES}
  )
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
  )
  set(primitiv_c_SRCS
    c/device.cc
    c/functions.cc
    c/graph.cc
    c/initializer.cc
    c/initializer_impl.cc
    c/internal.cc
    c/model.cc
    c/naive_device.cc
    c/optimizer.cc
    c/optimizer_impl.cc
    c/parameter.cc
    c/shape.cc
    c/status.cc
    c/tensor.cc
  )
  install(FILES ${primitiv_c_HDRS} DESTINATION include/primitiv/c)

  add_library(primitiv_c_OBJS OBJECT
    ${primitiv_c_HDRS}
    ${primitiv_c_SRCS}
  )
  set(primitiv_c_all_OBJS $<TARGET_OBJECTS:primitiv_c_OBJS>)

  if(PRIMITIV_USE_EIGEN)
    set(primitiv_c_eigen_HDRS c/eigen_device.h)
    set(primitiv_c_eigen_SRCS c/eigen_device.cc)
    install(FILES ${primitiv_c_eigen_HDRS} DESTINATION include/primitiv/c)

    add_library(primitiv_c_eigen_OBJS OBJECT
      ${primitiv_c_eigen_HDRS}
      ${primitiv_c_eigen_SRCS}
    )
    list(APPEND primitiv_c_all_OBJS $<TARGET_OBJECTS:primitiv_c_eigen_OBJS>)
  endif()

  if(PRIMITIV_USE_CUDA)
    set(primitiv_c_cuda_HDRS c/cuda_device.h)
    set(primitiv_c_cuda_SRCS c/cuda_device.cc)
    install(FILES ${primitiv_c_cuda_HDRS} DESTINATION include/primitiv/c)

    add_library(primitiv_c_cuda_OBJS OBJECT
      ${primitiv_c_cuda_HDRS}
      ${primitiv_c_cuda_SRCS}
    )
    list(APPEND primitiv_c_all_OBJS $<TARGET_OBJECTS:primitiv_c_cuda_OBJS>)
  endif()

  if(PRIMITIV_USE_OPENCL)
    set(primitiv_c_opencl_HDRS c/opencl_device.h)
    set(primitiv_c_opencl_SRCS c/opencl_device.cc)
    install(FILES ${primitiv_c_opencl_HDRS} DESTINATION include/primitiv/c)

    add_library(primitiv_c_opencl_OBJS OBJECT
      ${primitiv_c_opencl_HDRS}
      ${primitiv_c_opencl_SRCS}
    )
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
