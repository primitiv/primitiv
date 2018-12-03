# Build rules of the Eigen backend.

file(GLOB primitiv_c_eigen_HDRS "*.h")
file(GLOB primitiv_c_eigen_SRCS "*.cc")
install(
  FILES ${primitiv_c_eigen_HDRS}
  DESTINATION include/primitiv/c/devices/eigen)

add_library(primitiv_c_eigen_OBJS OBJECT
  ${primitiv_c_eigen_HDRS}
  ${primitiv_c_eigen_SRCS}
)
list(APPEND primitiv_c_all_OBJS $<TARGET_OBJECTS:primitiv_c_eigen_OBJS>)

set(primitiv_c_all_OBJS ${primitiv_c_all_OBJS} PARENT_SCOPE)
