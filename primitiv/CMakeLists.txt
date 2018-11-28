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

# Files on ./core
file(GLOB primitiv_core_HDRS "core/*.h")
file(GLOB primitiv_core_SRCS "core/*.cc")
install(FILES ${primitiv_core_HDRS} DESTINATION include/primitiv/core)

# Files on ./core/mixins
file(GLOB primitiv_core_mixins_HDRS "core/mixins/*.h")
file(GLOB primitiv_core_mixins_SRCS "core/mixins/*.cc")
install(
  FILES ${primitiv_core_mixins_HDRS} DESTINATION include/primitiv/core/mixins)

# Files on ./contrib
file(GLOB primitiv_contrib_HDRS "contrib/*.h")
file(GLOB primitiv_contrib_SRCS "contrib/*.cc")
install(FILES ${primitiv_contrib_HDRS} DESTINATION include/primitiv/contrib)

# Files on ./msgpack
file(GLOB primitiv_msgpack_HDRS "msgpack/*.h")
file(GLOB primitiv_msgpack_SRCS "msgpack/*.cc")
install(FILES ${primitiv_msgpack_HDRS} DESTINATION include/primitiv/msgpack)

# Files on ./devices/naive
file(GLOB primitiv_naive_HDRS "devices/naive/*.h")
file(GLOB primitiv_naive_SRCS "devices/naive/*.cc")
file(GLOB primitiv_naive_ops_HDRS "devices/naive/ops/*.h")
file(GLOB primitiv_naive_ops_SRCS "devices/naive/ops/*.cc")
install(FILES ${primitiv_naive_HDRS} DESTINATION include/primitiv/devices/naive)

# Minimal library: all of above.
set(primitiv_minimal_HDRS
  ${primitiv_core_HDRS}
  ${primitiv_core_mixins_HDRS}
  ${primitiv_contrib_HDRS}
  ${primitiv_msgpack_HDRS}
  ${primitiv_naive_HDRS}
  ${primitiv_naive_ops_HDRS}
)
set(primitiv_minimal_SRCS
  ${primitiv_core_SRCS}
  ${primitiv_core_mixins_SRCS}
  ${primitiv_contrib_SRCS}
  ${primitiv_msgpack_SRCS}
  ${primitiv_naive_SRCS}
  ${primitiv_naive_ops_SRCS}
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
  file(GLOB primitiv_eigen_HDRS "devices/eigen/*.h")
  file(GLOB primitiv_eigen_SRCS "devices/eigen/*.cc")
  file(GLOB primitiv_eigen_ops_HDRS "devices/eigen/ops/*.h")
  file(GLOB primitiv_eigen_ops_SRCS "devices/eigen/ops/*.cc")
  install(
    FILES ${primitiv_eigen_HDRS} DESTINATION include/primitiv/devices/eigen)

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
    ${primitiv_eigen_ops_SRCS}
    PROPERTIES COMPILE_FLAGS ${primitiv_eigen_COMPILE_FLAGS}
  )

  add_library(primitiv_eigen_OBJS OBJECT
    ${primitiv_core_HDRS}
    ${primitiv_eigen_HDRS}
    ${primitiv_eigen_SRCS}
    ${primitiv_eigen_ops_HDRS}
    ${primitiv_eigen_ops_SRCS}
  )

  list(APPEND primitiv_all_OBJS $<TARGET_OBJECTS:primitiv_eigen_OBJS>)
endif()

# Build rules of the CUDA backend.
if(PRIMITIV_USE_CUDA)
  # Common files
  file(GLOB primitiv_cuda_internal_HDRS "internal/cuda/*.h")
  file(GLOB primitiv_cuda_internal_SRCS "internal/cuda/*.cu")
  # devices::CUDA
  file(GLOB primitiv_cuda_HDRS "devices/cuda/*.h")
  file(GLOB primitiv_cuda_SRCS "devices/cuda/*.cu")
  file(GLOB primitiv_cuda_ops_HDRS "devices/cuda/ops/*.h")
  file(GLOB primitiv_cuda_ops_SRCS "devices/cuda/ops/*.cu")
  install(
    FILES ${primitiv_cuda_HDRS} DESTINATION include/primitiv/devices/cuda)
  # devices::CUDA16
  file(GLOB primitiv_cuda16_HDRS "devices/cuda16/*.h")
  file(GLOB primitiv_cuda16_SRCS "devices/cuda16/*.cu")
  file(GLOB primitiv_cuda16_ops_HDRS "devices/cuda16/ops/*.h")
  file(GLOB primitiv_cuda16_ops_SRCS "devices/cuda16/ops/*.cu")
  install(
    FILES ${primitiv_cuda16_HDRS} DESTINATION include/primitiv/devices/cuda16)

  list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -O3 -Xcompiler -fPIC")

  # Workaround for some systems.
  #list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES")
  #list(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED")

  # Supported architectures.
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
    ${primitiv_cuda_internal_HDRS}
    ${primitiv_cuda_internal_SRCS}
    ${primitiv_cuda_HDRS}
    ${primitiv_cuda_SRCS}
    ${primitiv_cuda_ops_HDRS}
    ${primitiv_cuda_ops_SRCS}
    ${primitiv_cuda16_HDRS}
    ${primitiv_cuda16_SRCS}
    ${primitiv_cuda16_ops_HDRS}
    ${primitiv_cuda16_ops_SRCS}
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
  # Common files
  file(GLOB primitiv_opencl_internal_HDRS "internal/opencl/*.h")
  # devices::OpenCL
  file(GLOB primitiv_opencl_HDRS "devices/opencl/*.h")
  file(GLOB primitiv_opencl_SRCS "devices/opencl/*.cc")
  file(GLOB primitiv_opencl_ops_SRCS "devices/opencl/ops/*.cc")
  file(GLOB primitiv_opencl_kernels_SRCS "devices/opencl/kernels/*.cl")
  list(REMOVE_ITEM primitiv_opencl_kernels_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/devices/opencl/kernels/common.cl")

  # Generates an embedded byte array of OpenCL kernels.
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/devices/opencl/kernels.inc
    COMMAND
      ${CMAKE_COMMAND}
      -P
      ${PROJECT_SOURCE_DIR}/cmake/generate_char_array.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/devices/opencl/kernels.inc
      ${CMAKE_CURRENT_SOURCE_DIR}/devices/opencl/kernels/common.cl
      ${primitiv_opencl_kernels_SRCS}
    DEPENDS
      ${primitiv_opencl_kernels_SRCS}
      ${CMAKE_CURRENT_SOURCE_DIR}/devices/opencl/kernels/common.cl
  )
  add_custom_target(primitiv_opencl_KERNELS DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/devices/opencl/kernels.inc
  )

  install(
    FILES ${primitiv_opencl_HDRS} DESTINATION include/primitiv/devices/opencl)

  add_library(primitiv_opencl_OBJS OBJECT
    ${primitiv_core_HDRS}
    ${primitiv_opencl_internal_HDRS}
    ${primitiv_opencl_HDRS}
    ${primitiv_opencl_SRCS}
    ${primitiv_opencl_ops_SRCS}
  )
  add_dependencies(primitiv_opencl_OBJS primitiv_opencl_KERNELS)

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
  file(GLOB primitiv_c_HDRS "c/*.h")
  file(GLOB primitiv_c_SRCS "c/*.cc")
  file(GLOB primitiv_c_internal_HDRS "c/internal/*.h")
  file(GLOB primitiv_c_internal_SRCS "c/internal/*.cc")
  install(FILES ${primitiv_c_HDRS} DESTINATION include/primitiv/c)

  file(GLOB primitiv_c_naive_HDRS "c/devices/naive/*.h")
  file(GLOB primitiv_c_naive_SRCS "c/devices/naive/*.cc")
  install(
    FILES ${primitiv_c_naive_HDRS}
    DESTINATION include/primitiv/c/devices/naive)

  add_library(primitiv_c_OBJS OBJECT
    ${primitiv_c_HDRS}
    ${primitiv_c_SRCS}
    ${primitiv_c_internal_HDRS}
    ${primitiv_c_internal_SRCS}
    ${primitiv_c_naive_HDRS}
    ${primitiv_c_naive_SRCS}
  )
  set(primitiv_c_all_OBJS $<TARGET_OBJECTS:primitiv_c_OBJS>)

  if(PRIMITIV_USE_EIGEN)
    file(GLOB primitiv_c_eigen_HDRS "c/devices/eigen/*.h")
    file(GLOB primitiv_c_eigen_SRCS "c/devices/eigen/*.cc")
    install(
      FILES ${primitiv_c_eigen_HDRS}
      DESTINATION include/primitiv/c/devices/eigen)

    add_library(primitiv_c_eigen_OBJS OBJECT
      ${primitiv_c_eigen_HDRS}
      ${primitiv_c_eigen_SRCS}
    )
    list(APPEND primitiv_c_all_OBJS $<TARGET_OBJECTS:primitiv_c_eigen_OBJS>)
  endif()

  if(PRIMITIV_USE_CUDA)
    file(GLOB primitiv_c_cuda_HDRS "c/devices/cuda/*.h")
    file(GLOB primitiv_c_cuda_SRCS "c/devices/cuda/*.cc")
    install(
      FILES ${primitiv_c_cuda_HDRS}
      DESTINATION include/primitiv/c/devices/cuda)

    add_library(primitiv_c_cuda_OBJS OBJECT
      ${primitiv_c_cuda_HDRS}
      ${primitiv_c_cuda_SRCS}
    )
    list(APPEND primitiv_c_all_OBJS $<TARGET_OBJECTS:primitiv_c_cuda_OBJS>)
  endif()

  if(PRIMITIV_USE_OPENCL)
    file(GLOB primitiv_c_opencl_HDRS "c/devices/opencl/*.h")
    file(GLOB primitiv_c_opencl_SRCS "c/devices/opencl/*.cc")
    install(
      FILES ${primitiv_c_opencl_HDRS}
      DESTINATION include/primitiv/c/devices/opencl)

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
