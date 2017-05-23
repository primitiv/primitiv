set(nmtkit_LIBRARY_HDRS
  constant_initializer.h
  cpu_device.h
  device.h
  function.h
  function_impl.h
  graph.h
  initializer.h
  node.h
  node_ops.h
  parameter.h
  sgd_trainer.h
  shape.h
  tensor.h
  tensor_ops.h
  trainer.h
)

set(nmtkit_LIBRARY_SRCS
  constant_initializer.cc
  cpu_device.cc
  function_impl.cc
  graph.cc
  parameter.cc
  sgd_trainer.cc
  shape.cc
  tensor.cc
)

add_library(primitiv ${nmtkit_LIBRARY_HDRS} ${nmtkit_LIBRARY_SRCS})
target_link_libraries(primitiv)
