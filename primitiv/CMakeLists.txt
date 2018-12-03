# Build rules of the primitiv core library.

# Global configuration header
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
)

# Version header.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/version.h
  @ONLY
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
  ${CMAKE_CURRENT_BINARY_DIR}/version.h
  DESTINATION include/primitiv
)

# Combined header (won't be used by the library itself)
install(FILES primitiv.h DESTINATION include/primitiv)

# Files on ./core
file(GLOB primitiv_core_HDRS "core/*.h")
file(GLOB primitiv_core_SRCS "core/*.cc")
install(FILES ${primitiv_core_HDRS} DESTINATION include/primitiv/core)

# Files on ./core/mixins
file(GLOB primitiv_core_mixins_HDRS "core/mixins/*.h")
file(GLOB primitiv_core_mixins_SRCS "core/mixins/*.cc")
install(
  FILES ${primitiv_core_mixins_HDRS} DESTINATION include/primitiv/core/mixins)

# Files on ./contrib
file(GLOB primitiv_contrib_HDRS "contrib/*.h")
file(GLOB primitiv_contrib_SRCS "contrib/*.cc")
install(FILES ${primitiv_contrib_HDRS} DESTINATION include/primitiv/contrib)

# Files on ./msgpack
file(GLOB primitiv_msgpack_HDRS "msgpack/*.h")
file(GLOB primitiv_msgpack_SRCS "msgpack/*.cc")
install(FILES ${primitiv_msgpack_HDRS} DESTINATION include/primitiv/msgpack)

# Minimal library: all of above.
set(primitiv_minimal_HDRS
  ${primitiv_core_HDRS}
  ${primitiv_core_mixins_HDRS}
  ${primitiv_contrib_HDRS}
  ${primitiv_msgpack_HDRS}
)
set(primitiv_minimal_SRCS
  ${primitiv_core_SRCS}
  ${primitiv_core_mixins_SRCS}
  ${primitiv_contrib_SRCS}
  ${primitiv_msgpack_SRCS}
)

# Builds the minimal library.
add_library(primitiv_core_OBJS OBJECT
  ${primitiv_minimal_HDRS}
  ${primitiv_minimal_SRCS}
)
set(primitiv_all_OBJS $<TARGET_OBJECTS:primitiv_core_OBJS>)
set(primitiv_all_DEPS)

# devices
add_subdirectory(devices)

# Builds the integrated binary.
if(PRIMITIV_BUILD_STATIC_LIBRARY)
  add_library(primitiv STATIC ${primitiv_all_OBJS})
else()
  add_library(primitiv SHARED ${primitiv_all_OBJS})
endif()
target_link_libraries(primitiv ${primitiv_all_DEPS})

install(TARGETS primitiv DESTINATION lib)

# Builds C API
if(PRIMITIV_BUILD_C_API)
  file(GLOB primitiv_c_HDRS "c/*.h")
  file(GLOB primitiv_c_SRCS "c/*.cc")
  file(GLOB primitiv_c_internal_HDRS "c/internal/*.h")
  file(GLOB primitiv_c_internal_SRCS "c/internal/*.cc")
  install(FILES ${primitiv_c_HDRS} DESTINATION include/primitiv/c)

  add_library(primitiv_c_OBJS OBJECT
    ${primitiv_c_HDRS}
    ${primitiv_c_SRCS}
    ${primitiv_c_internal_HDRS}
    ${primitiv_c_internal_SRCS}
  )
  set(primitiv_c_all_OBJS $<TARGET_OBJECTS:primitiv_c_OBJS>)

  # devices
  add_subdirectory(c/devices)

  if(PRIMITIV_BUILD_STATIC_LIBRARY)
    add_library(primitiv_c STATIC ${primitiv_c_all_OBJS})
  else()
    add_library(primitiv_c SHARED ${primitiv_c_all_OBJS})
  endif()
  target_link_libraries(primitiv_c primitiv)

  install(TARGETS primitiv_c DESTINATION lib)
endif()
