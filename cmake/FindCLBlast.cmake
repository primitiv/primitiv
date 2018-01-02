# - Try to find CLBlast lib
# Once done this will define
#
#  CLBlast_FOUND - system has CLBlast lib with correct version
#  CLBLAST_INCLUDE_DIRS - CLBlast include directory
#  CLBLAST_LIBRARIES - CLBlast libraries
#

if(NOT CLBLAST_ROOT_DIR)
    find_path(CLBLAST_ROOT_DIR
        NAMES include/clblast.h
        HINTS ${CMAKE_INSTALL_PREFIX})
endif(NOT CLBLAST_ROOT_DIR)

if(NOT CLBLAST_INCLUDE_DIRS)
    find_path(CLBLAST_INCLUDE_DIRS
        NAMES clblast.h
        HINTS ${CLBLAST_ROOT_DIR}/include)
endif(NOT CLBLAST_INCLUDE_DIRS)

if(NOT CLBLAST_LIBRARIES)
    find_library(CLBLAST_LIBRARIES
        NAMES libclblast.so
        HINTS ${CLBLAST_ROOT_DIR}/lib ${CLBLAST_ROOT_DIR}/lib64 ${CLBLAST_ROOT_DIR}/lib32)
endif(NOT CLBLAST_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CLBlast DEFAULT_MSG CLBLAST_INCLUDE_DIRS CLBLAST_LIBRARIES CLBLAST_ROOT_DIR)

mark_as_advanced(CLBLAST_INCLUDE_DIRS CLBLAST_LIBRARIES)
