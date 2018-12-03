# Build rules of the CUDA backend.

file(GLOB primitiv_c_cuda_HDRS "*.h")
file(GLOB primitiv_c_cuda_SRCS "*.cc")
install(
  FILES ${primitiv_c_cuda_HDRS}
  DESTINATION include/primitiv/c/devices/cuda)

add_library(primitiv_c_cuda_OBJS OBJECT
  ${primitiv_c_cuda_HDRS}
  ${primitiv_c_cuda_SRCS}
)
list(APPEND primitiv_c_all_OBJS $<TARGET_OBJECTS:primitiv_c_cuda_OBJS>)

set(primitiv_c_all_OBJS ${primitiv_c_all_OBJS} PARENT_SCOPE)
