# Build rules of the Naive backend.

file(GLOB primitiv_naive_HDRS "*.h")
file(GLOB primitiv_naive_SRCS "*.cc")
file(GLOB primitiv_naive_ops_HDRS "ops/*.h")
file(GLOB primitiv_naive_ops_SRCS "ops/*.cc")
install(FILES ${primitiv_naive_HDRS} DESTINATION include/primitiv/devices/naive)

set(primitiv_naive_HDRS
  ${primitiv_core_HDRS}
  ${primitiv_naive_HDRS}
  ${primitiv_naive_ops_HDRS}
)
set(primitiv_naive_SRCS
  ${primitiv_naive_SRCS}
  ${primitiv_naive_ops_SRCS}
)

add_library(primitiv_naive_OBJS OBJECT
  ${primitiv_naive_HDRS}
  ${primitiv_naive_SRCS}
)
list(APPEND primitiv_all_OBJS $<TARGET_OBJECTS:primitiv_naive_OBJS>)

set(primitiv_all_OBJS ${primitiv_all_OBJS} PARENT_SCOPE)
