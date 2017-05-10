set(nmtkit_LIBRARY_HDRS
  cpu_device.h
  device.h
  shape.h
  tensor.h
)

set(nmtkit_LIBRARY_SRCS
  cpu_device.cc
  shape.cc
  tensor.cc
)

add_library(primitiv ${nmtkit_LIBRARY_HDRS} ${nmtkit_LIBRARY_SRCS})
target_link_libraries(primitiv)
