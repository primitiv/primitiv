# OpenCL tuner definitions

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

function(primitiv_tuner name)
  add_executable(${name}_tuner
    common.h parameter_tuner.h ${name}_tuner.cc)
  target_link_libraries(${name}_tuner primitiv)
endfunction()

primitiv_tuner(set_identity)
primitiv_tuner(pick_fw)
primitiv_tuner(concat_fw)
primitiv_tuner(slice_fw)
primitiv_tuner(pick_bw)
primitiv_tuner(flip_fw)
