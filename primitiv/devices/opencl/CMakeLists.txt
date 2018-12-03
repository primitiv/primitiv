# Build rules of the OpenCL backend.

# Common files
file(GLOB primitiv_opencl_internal_HDRS "../../internal/opencl/*.h")
# devices::OpenCL
file(GLOB primitiv_opencl_HDRS "*.h")
file(GLOB primitiv_opencl_SRCS "*.cc")
file(GLOB primitiv_opencl_ops_SRCS "ops/*.cc")
file(GLOB primitiv_opencl_kernels_SRCS "kernels/*.cl")
list(REMOVE_ITEM primitiv_opencl_kernels_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/common.cl")

# Generates an embedded byte array of OpenCL kernels.
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernels.inc
  COMMAND
    ${CMAKE_COMMAND}
    -P
    ${PROJECT_SOURCE_DIR}/cmake/generate_char_array.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/kernels.inc
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common.cl
    ${primitiv_opencl_kernels_SRCS}
  DEPENDS
    ${primitiv_opencl_kernels_SRCS}
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common.cl
)
add_custom_target(primitiv_opencl_KERNELS DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/kernels.inc
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

set(primitiv_all_OBJS ${primitiv_all_OBJS} PARENT_SCOPE)
set(primitiv_all_DEPS ${primitiv_all_DEPS} PARENT_SCOPE)
