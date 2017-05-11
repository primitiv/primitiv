#include <config.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <primitiv/cpu_device.h>

using std::cerr;
using std::endl;

namespace primitiv {

CPUDevice::~CPUDevice() {
  // check memory leak
  if (!blocks_.empty()) {
    cerr << "FATAL ERROR: Detected memory leak on CPUDevice!" << endl;
    cerr << "Leaked blocks (handle: size):" << endl;
    for (const auto &kv : blocks_) {
      cerr << "  " << kv.first << ": " << kv.second << endl;
    }
    std::abort();
  }
}

void *CPUDevice::allocate(const unsigned size) {
  if (size == 0) {
    throw std::runtime_error("Attempted to allocate a zero-size memory.");
  }

  void *ptr = std::malloc(size);

  if (!ptr) {
    std::stringstream ss;
    ss << "Memory allocation failed. Requested size: " << size;
    throw std::runtime_error(ss.str());
  }

  blocks_.insert(std::make_pair(ptr, size));
  return ptr;
}

void CPUDevice::free(void *ptr) {
  if (ptr == nullptr) return;

  auto it = blocks_.find(ptr);
  if (it == blocks_.end()) {
    std::stringstream ss;
    ss << "Attempted to dispose unknown memory block: " << ptr;
    throw std::runtime_error(ss.str());
  }
  blocks_.erase(it);

  std::free(ptr);
}

void CPUDevice::copy_to_device(
    void *dest, const void *src, const unsigned size) {
  std::memcpy(dest, src, size);
}

void CPUDevice::copy_to_host(
    void *dest, const void *src, const unsigned size) {
  std::memcpy(dest, src, size);
}

#define CHECK_DEVICE(x) { \
  if ((x).device().get() != this) { \
    std::stringstream ss; \
    ss << "Device mismatched. (" #x ").device(): " << (x).device().get() \
       << "!= this:" << this; \
    throw std::runtime_error(ss.str()); \
  } \
}

Tensor CPUDevice::add_const(const Tensor &x, const float k) {
  CHECK_DEVICE(x);

  Tensor ret(x.shape(), x.device());
  float *dest = static_cast<float *>(ret.data());
  const float *src = static_cast<const float *>(x.data());
  const unsigned size = x.shape().size();
  for (unsigned i = 0; i < size; ++i) {
    dest[i] = src[i] + k;
  }
  return ret;
}

}  // namespace primitiv
