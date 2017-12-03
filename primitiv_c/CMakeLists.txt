set(primitiv_c_HDRS
  api.h
  device.h
  graph.h
  initializer.h
  internal.h
  model.h
  optimizer.h
  parameter.h
  shape.h
  tensor.h
)
set(primitiv_c_SRCS
  device.cc
  graph.cc
  initializer.cc
  model.cc
  optimizer.cc
  parameter.cc
  shape.cc
  tensor.cc
)
install(FILES ${primitiv_c_HDRS} DESTINATION include/primitiv_c)

set(BUILD_SHARED_LIBS ON)

add_library(primitiv_c ${primitiv_c_HDRS} ${primitiv_c_SRCS})
target_link_libraries(primitiv_c primitiv)

install(TARGETS primitiv_c DESTINATION lib)