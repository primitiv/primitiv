# Build rules of the OpenCL backend.

file(GLOB primitiv_c_opencl_HDRS "*.h")
file(GLOB primitiv_c_opencl_SRCS "*.cc")
install(
  FILES ${primitiv_c_opencl_HDRS}
  DESTINATION include/primitiv/c/devices/opencl)

add_library(primitiv_c_opencl_OBJS OBJECT
  ${primitiv_c_opencl_HDRS}
  ${primitiv_c_opencl_SRCS}
)
list(APPEND primitiv_c_all_OBJS $<TARGET_OBJECTS:primitiv_c_opencl_OBJS>)

set(primitiv_c_all_OBJS ${primitiv_c_all_OBJS} PARENT_SCOPE)
