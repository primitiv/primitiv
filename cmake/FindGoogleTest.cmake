# Try to find GoogleTest library.
#
# This script will use:
#
#   GOOGLETEST_INCLUDE_DIR - User-defined path of the GoogleTest include
#                            directory.
#   GOOGLETEST_LIBRARY_DIR - User-defined path of the GoogleTest libraries.
#
# and once done this script will define:
#
#   GOOGLETEST_FOUND       - System has GoogleTest libraries.
#   GOOGLETEST_INCLUDE_DIR - The GoogleTest include directory.
#   GOOGLETEST_LIBRARIES   - The GoogleTest library directory.

if(NOT GOOGLETEST_INCLUDE_DIR)
  find_path(GOOGLETEST_INCLUDE_DIR NAMES gtest/gtest.h)
endif()

find_library(GOOGLETEST_gtest_LIBRARY
  NAMES gtest HINTS ${GOOGLETEST_LIBRARY_DIR})
find_library(GOOGLETEST_gtest_main_LIBRARY
  NAMES gtest_main HINTS ${GOOGLETEST_LIBRARY_DIR})


if (GOOGLETEST_INCLUDE_DIR AND
    GOOGLETEST_gtest_LIBRARY AND
    GOOGLETEST_gtest_main_LIBRARY)
  set(GOOGLETEST_FOUND ON)
  set(GOOGLETEST_LIBRARIES
    ${GOOGLETEST_gtest_LIBRARY}
    ${GOOGLETEST_gtest_main_LIBRARY})
else()
  message(SEND_ERROR "GoogleTest not found.")
endif()
