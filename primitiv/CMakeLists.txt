set(primitiv_base_HDRS
  cpu_device.h
  device.h
  error.h
  function.h
  function_impl.h
  graph.h
  initializer.h
  initializer_impl.h
  node_ops.h
  parameter.h
  primitiv.h
  primitiv_cuda.h
  shape.h
  shape_ops.h
  tensor.h
  tensor_ops.h
  trainer.h
  trainer_impl.h)

set(primitiv_base_SRCS
  cpu_device.cc
  device.cc
  function_impl.cc
  graph.cc
  initializer_impl.cc
  node_ops.cc
  parameter.cc
  shape.cc
  shape_ops.cc
  tensor.cc
  tensor_ops.cc
  trainer.cc
  trainer_impl.cc)

protobuf_generate_cpp(primitiv_proto_SRCS primitiv_proto_HDRS messages.proto)

include_directories(${PROJECT_SOURCE_DIR}/submodules/yaml-cpp/include)
link_directories(${PROJECT_BINARY_DIR}/submodules/yaml-cpp)

add_library(primitiv_base OBJECT
  ${primitiv_base_HDRS} ${primitiv_base_SRCS}
  ${primitiv_proto_HDRS} ${primitiv_proto_SRCS})
set(primitiv_OBJS $<TARGET_OBJECTS:primitiv_base>)
set(primitiv_DEPS protobuf)
set(primitiv_HDRS ${primitiv_base_HDRS})

if(PRIMITIV_USE_CUDA)
  set(primitiv_cuda_HDRS
    cuda_device.h
    cuda_memory_pool.h
    cuda_utils.h)
  set(primitiv_cuda_SRCS
    cuda_device.cu
    cuda_memory_pool.cu
    cuda_utils.cu)
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

  cuda_compile(primitiv_cuda_OBJ
    ${primitiv_base_HDRS}
    ${primitiv_cuda_HDRS}
    ${primitiv_cuda_SRCS})
  list(APPEND primitiv_OBJS ${primitiv_cuda_OBJ})
  list(APPEND primitiv_DEPS
    ${CUDA_LIBRARIES}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_curand_LIBRARY})
  list(APPEND primitiv_HDRS ${primitiv_cuda_HDRS})
endif()

if(PRIMITIV_BUILD_STATIC_LIBRARY)
  add_library(primitiv STATIC ${primitiv_OBJS})
else()
  add_library(primitiv SHARED ${primitiv_OBJS})
endif()
target_link_libraries(primitiv ${primitiv_DEPS})

install(TARGETS primitiv DESTINATION lib)
install(FILES ${primitiv_HDRS} DESTINATION include/primitiv)
