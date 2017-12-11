if (PRIMITIV_INCLUDE_DIR AND PRIMITIV_LIBRARY_DIR)
  # in cache already
  set(PRIMITIV_FOUND ON)
else ()
  # search first if an PrimitivConfig.cmake is available in the system,
  # if successful this would set PRIMITIV_INCLUDE_DIR and PRIMITIV_LIBRARY_DIR
  # and the rest of the script will work as usual
  find_package(Primitiv NO_MODULE QUIET)

  if (NOT PRIMITIV_INCLUDE_DIR)
    find_path(PRIMITIV_INCLUDE_DIR
      NAMES primitiv/primitiv.h
        HINTS
          ENV PRIMITIV_ROOT
          ENV PRIMITIV_ROOT_DIR
      PATHS
        ${CMAKE_INSTALL_PREFIX}/include
    )
  endif(NOT PRIMITIV_INCLUDE_DIR)

  if (NOT PRIMITIV_LIBRARY_DIR)
    find_path(PRIMITIV_LIBRARY_DIR
      NAMES libprimitiv.so libprimitiv.dylib
        HINTS
          $ENV{PRIMITIV_ROOT}/build/primitiv
          $ENV{PRIMITIV_ROOT_DIR}/build/primitiv
      PATHS
        ${CMAKE_INSTALL_PREFIX}/lib
    )
  endif(NOT PRIMITIV_LIBRARY_DIR)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Primitiv DEFAULT_MSG PRIMITIV_INCLUDE_DIR PRIMITIV_LIBRARY_DIR)
  mark_as_advanced(PRIMITIV_INCLUDE_DIR)
  mark_as_advanced(PRIMITIV_LIBRARY_DIR)

  if (PRIMITIV_INCLUDE_DIR AND PRIMITIV_LIBRARY_DIR)
    message("-- Found Primitiv: " ${PRIMITIV_LIBRARY_DIR})
    # message("-- Primitiv include dir: " ${PRIMITIV_INCLUDE_DIR})
    # message("-- Primitiv libraries: " ${PRIMITIV_LIBRARY_DIR})
    set(PRIMITIV_FOUND ON)
  else ()
    message(STATUS "Primitiv is not installed.")
    set(PRIMITIV_FOUND OFF)
  endif()
endif()
