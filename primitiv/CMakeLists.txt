set(nmtkit_LIBRARY_HDRS
  cpu_device.h
  device.h
  function.h
  function_impl.h
  graph.h
  node.h
  shape.h
  tensor.h
  tensor_ops.h
)

set(nmtkit_LIBRARY_SRCS
  cpu_device.cc
  function_impl.cc
  graph.cc
  shape.cc
  tensor.cc
)

add_library(primitiv ${nmtkit_LIBRARY_HDRS} ${nmtkit_LIBRARY_SRCS})
target_link_libraries(primitiv)
