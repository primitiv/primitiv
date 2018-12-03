# Build rules of the Naive backend.

file(GLOB primitiv_c_naive_HDRS "*.h")
file(GLOB primitiv_c_naive_SRCS "*.cc")
install(
  FILES ${primitiv_c_naive_HDRS}
  DESTINATION include/primitiv/c/devices/naive)

add_library(primitiv_c_naive_OBJS OBJECT
  ${primitiv_c_naive_HDRS}
  ${primitiv_c_naive_SRCS}
)
list(APPEND primitiv_c_all_OBJS $<TARGET_OBJECTS:primitiv_c_naive_OBJS>)

set(primitiv_c_all_OBJS ${primitiv_c_all_OBJS} PARENT_SCOPE)
