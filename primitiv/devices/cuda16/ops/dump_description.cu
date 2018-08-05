#include <primitiv/config.h>

#include <iostream>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace primitiv {
namespace devices {

void CUDA16::dump_description() const {
  using std::cerr;
  using std::endl;

  cerr << "Device " << this << endl;
  cerr << "  Type: CUDA16" << endl;

  const ::cudaDeviceProp &prop = state_->prop;
  cerr << "  Device ID: " << dev_id_ << endl;
  cerr << "    Name .................. " << prop.name << endl;
  cerr << "    Global memory ......... " << prop.totalGlobalMem << endl;
  cerr << "    Shared memory/block ... " << prop.sharedMemPerBlock << endl;
  cerr << "    Threads/block ......... " << prop.maxThreadsPerBlock << endl;
  cerr << "    Block size ............ " << prop.maxThreadsDim[0] << ", "
                                         << prop.maxThreadsDim[1] << ", "
                                         << prop.maxThreadsDim[2] << endl;
  cerr << "    Grid size ............. " << prop.maxGridSize[0] << ", "
                                         << prop.maxGridSize[1] << ", "
                                         << prop.maxGridSize[2] << endl;
  cerr << "    Compute capability .... " << prop.major << '.'
                                         << prop.minor << endl;
  /*
  cerr << "  Configurations:" << endl;
  cerr << "    1 dim ........... " << dim1_x_ << " threads" << endl;
  cerr << "    2 dims .......... " << dim2_x_ << "x"
                                   << dim2_y_ << " threads" << endl;
  cerr << "    Maximum batch ... " << max_batch_ << endl;
  */
}

}  // namespace devices
}  // namespace primitiv
