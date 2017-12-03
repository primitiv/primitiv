find_path(PRIMITIV_INCLUDE_DIR NAMES primitiv/primitiv.h
  HINTS
  ENV PRIMITIV_ROOT
  ENV PRIMITIV_ROOT_DIR
  PATHS
  ${CMAKE_INSTALL_PREFIX}/include
)

find_library(PRIMITIV_LIBRARIES NAMES primitiv
  HINTS
  ENV PRIMITIV_ROOT
  ENV PRIMITIV_ROOT_DIR
  PATHS
  ${CMAKE_INSTALL_PREFIX}/lib
)

if (PRIMITIV_INCLUDE_DIR AND PRIMITIV_LIBRARIES)
  message("-- Found Primitiv: " ${PRIMITIV_LIBRARIES})
  #message("-- Primitiv include dir: " ${PRIMITIV_INCLUDE_DIR})
  #message("-- Primitiv libraries: " ${PRIMITIV_LIBRARIES})
  set(PRIMITIV_FOUND ON)
else()
  message(STATUS "Primitiv is not installed.")
  set(PRIMITIV_FOUND OFF)
endif()