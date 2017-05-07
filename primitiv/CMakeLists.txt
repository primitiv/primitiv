set(nmtkit_LIBRARY_HDRS
  shape.h
)

set(nmtkit_LIBRARY_SRCS
  shape.cc
)

add_library(primitiv ${nmtkit_LIBRARY_HDRS} ${nmtkit_LIBRARY_SRCS})
target_link_libraries(primitiv)
