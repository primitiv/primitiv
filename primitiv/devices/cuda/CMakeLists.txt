# Build rules of the CUDA backend.

# Common files
file(GLOB primitiv_cuda_internal_HDRS "../../internal/cuda/*.h")
file(GLOB primitiv_cuda_internal_SRCS "../../internal/cuda/*.cu")
# devices::CUDA
file(GLOB primitiv_cuda_HDRS "*.h")
file(GLOB primitiv_cuda_SRCS "*.cu")
file(GLOB primitiv_cuda_ops_HDRS "ops/*.h")
file(GLOB primitiv_cuda_ops_SRCS "ops/*.cu")
install(
  FILES ${primitiv_cuda_HDRS} DESTINATION include/primitiv/devices/cuda)

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
)
list(APPEND primitiv_all_OBJS ${primitiv_cuda_OBJS})
list(APPEND primitiv_all_DEPS
  ${CUDA_LIBRARIES}
  ${CUDA_cublas_LIBRARY}
  ${CUDA_curand_LIBRARY}
  ${CUDNN_LIBRARIES}
)

set(primitiv_all_OBJS ${primitiv_all_OBJS} PARENT_SCOPE)
set(primitiv_all_DEPS ${primitiv_all_DEPS} PARENT_SCOPE)
