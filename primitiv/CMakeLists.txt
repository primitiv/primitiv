set(nmtkit_LIBRARY_HDRS
  shape.h
  tensor.h
)

set(nmtkit_LIBRARY_SRCS
  shape.cc
  tensor.cc
)

add_library(primitiv ${nmtkit_LIBRARY_HDRS} ${nmtkit_LIBRARY_SRCS})
target_link_libraries(primitiv)
