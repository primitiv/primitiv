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

#define CHECK_DEVICE(x) \
  if ((x).device().get() != this) { \
    std::stringstream ss; \
    ss << "Device mismatched. (" #x ").device(): " << (x).device().get() \
       << "!= this:" << this; \
    throw std::runtime_error(ss.str()); \
  }

#define DATA(x) static_cast<float *>((x).data());
#define CDATA(x) static_cast<const float *>((x).data());

#define REPEAT_OP(i, n, dest, op) \
  for (unsigned (i) = 0; (i) < (n); ++(i)) { \
    (dest) = (op); \
  }

Tensor CPUDevice::add(const Tensor &x, const float k) {
  CHECK_DEVICE(x);

  Tensor ret(x.shape(), x.device());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i], src[i] + k);
  return ret;
}

Tensor CPUDevice::add(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const float *src_a = CDATA(a);
  const float *src_b = CDATA(b);

  if (sa.dims() == sb.dims()) {
    if (sa.batch_size() == sb.batch_size()) {
      // ret = a + b
      Tensor ret(sa, a.device());
      float *dest = DATA(ret);
      const unsigned size = sa.size();
      REPEAT_OP(i, size, dest[i], src_a[i] + src_b[i]);
      return ret;
    } else if (sa.batch_size() == 1) {
      // ret = batch_broadcast(a) + b
      Tensor ret(sb, a.device());
      float *dest = DATA(ret);
      const unsigned ms = sa.size();
      const unsigned bs = sb.batch_size();
      for (unsigned k = 0; k < bs; ++k, dest += ms, src_b += ms) {
        REPEAT_OP(i, ms, dest[i], src_a[i] + src_b[i]);
      }
      return ret;
    } else if (sb.batch_size() == 1) {
      // ret = a + batch_broadcast(b)
      Tensor ret(sa, a.device());
      float *dest = DATA(ret);
      const unsigned ms = sb.size();
      const unsigned bs = sa.batch_size();
      for (unsigned k = 0; k < bs; ++k, dest += ms, src_a += ms) {
        REPEAT_OP(i, ms, dest[i], src_a[i] + src_b[i]);
      }
      return ret;
    }
  }

  // error
  std::stringstream ss;
  ss << "Attempted to add tensors with shapes "
     << a.shape().to_string() << " and " << b.shape().to_string() << '.';
  throw std::runtime_error(ss.str());
}

Tensor CPUDevice::subtract(const Tensor &x, const float k) {
  CHECK_DEVICE(x);

  Tensor ret(x.shape(), x.device());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i], src[i] - k);
  return ret;
}

Tensor CPUDevice::subtract(const float k, const Tensor &x) {
  CHECK_DEVICE(x);

  Tensor ret(x.shape(), x.device());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i], k - src[i]);
  return ret;
}

Tensor CPUDevice::subtract(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const float *src_a = CDATA(a);
  const float *src_b = CDATA(b);

  if (sa.dims() == sb.dims()) {
    if (sa.batch_size() == sb.batch_size()) {
      // ret = a - b
      Tensor ret(sa, a.device());
      float *dest = DATA(ret);
      const unsigned size = sa.size();
      REPEAT_OP(i, size, dest[i], src_a[i] - src_b[i]);
      return ret;
    } else if (sa.batch_size() == 1) {
      // ret = batch_broadcast(a) - b
      Tensor ret(sb, a.device());
      float *dest = DATA(ret);
      const unsigned ms = sa.size();
      const unsigned bs = sb.batch_size();
      for (unsigned k = 0; k < bs; ++k, dest += ms, src_b += ms) {
        REPEAT_OP(i, ms, dest[i], src_a[i] - src_b[i]);
      }
      return ret;
    } else if (sb.batch_size() == 1) {
      // ret = a - batch_broadcast(b)
      Tensor ret(sa, a.device());
      float *dest = DATA(ret);
      const unsigned ms = sb.size();
      const unsigned bs = sa.batch_size();
      for (unsigned k = 0; k < bs; ++k, dest += ms, src_a += ms) {
        REPEAT_OP(i, ms, dest[i], src_a[i] - src_b[i]);
      }
      return ret;
    }
  }

  // error
  std::stringstream ss;
  ss << "Attempted to subtract tensors with shapes "
     << a.shape().to_string() << " and " << b.shape().to_string() << '.';
  throw std::runtime_error(ss.str());
}

Tensor CPUDevice::multiply(const Tensor &x, const float k) {
  CHECK_DEVICE(x);

  Tensor ret(x.shape(), x.device());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i], src[i] * k);
  return ret;
}

Tensor CPUDevice::multiply(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const float *src_a = CDATA(a);
  const float *src_b = CDATA(b);

  if (sa.dims() == sb.dims()) {
    if (sa.batch_size() == sb.batch_size()) {
      // ret = a * b
      Tensor ret(sa, a.device());
      float *dest = DATA(ret);
      const unsigned size = sa.size();
      REPEAT_OP(i, size, dest[i], src_a[i] * src_b[i]);
      return ret;
    } else if (sa.batch_size() == 1) {
      // ret = batch_broadcast(a) * b
      Tensor ret(sb, a.device());
      float *dest = DATA(ret);
      const unsigned ms = sa.size();
      const unsigned bs = sb.batch_size();
      for (unsigned k = 0; k < bs; ++k, dest += ms, src_b += ms) {
        REPEAT_OP(i, ms, dest[i], src_a[i] * src_b[i]);
      }
      return ret;
    } else if (sb.batch_size() == 1) {
      // ret = a * batch_broadcast(b)
      Tensor ret(sa, a.device());
      float *dest = DATA(ret);
      const unsigned ms = sb.size();
      const unsigned bs = sa.batch_size();
      for (unsigned k = 0; k < bs; ++k, dest += ms, src_a += ms) {
        REPEAT_OP(i, ms, dest[i], src_a[i] * src_b[i]);
      }
      return ret;
    }
  }

  // error
  std::stringstream ss;
  ss << "Attempted to multiply tensors with shapes "
     << a.shape().to_string() << " and " << b.shape().to_string() << '.';
  throw std::runtime_error(ss.str());
}

Tensor CPUDevice::divide(const Tensor &x, const float k) {
  CHECK_DEVICE(x);

  Tensor ret(x.shape(), x.device());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i], src[i] / k);
  return ret;
}

Tensor CPUDevice::divide(const float k, const Tensor &x) {
  CHECK_DEVICE(x);

  Tensor ret(x.shape(), x.device());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i], k / src[i]);
  return ret;
}

Tensor CPUDevice::divide(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const float *src_a = CDATA(a);
  const float *src_b = CDATA(b);

  if (sa.dims() == sb.dims()) {
    if (sa.batch_size() == sb.batch_size()) {
      // ret = a / b
      Tensor ret(sa, a.device());
      float *dest = DATA(ret);
      const unsigned size = sa.size();
      REPEAT_OP(i, size, dest[i], src_a[i] / src_b[i]);
      return ret;
    } else if (sa.batch_size() == 1) {
      // ret = batch_broadcast(a) / b
      Tensor ret(sb, a.device());
      float *dest = DATA(ret);
      const unsigned ms = sa.size();
      const unsigned bs = sb.batch_size();
      for (unsigned k = 0; k < bs; ++k, dest += ms, src_b += ms) {
        REPEAT_OP(i, ms, dest[i], src_a[i] / src_b[i]);
      }
      return ret;
    } else if (sb.batch_size() == 1) {
      // ret = a / batch_broadcast(b)
      Tensor ret(sa, a.device());
      float *dest = DATA(ret);
      const unsigned ms = sb.size();
      const unsigned bs = sa.batch_size();
      for (unsigned k = 0; k < bs; ++k, dest += ms, src_a += ms) {
        REPEAT_OP(i, ms, dest[i], src_a[i] / src_b[i]);
      }
      return ret;
    }
  }

  // error
  std::stringstream ss;
  ss << "Attempted to divide tensors with shapes "
     << a.shape().to_string() << " and " << b.shape().to_string() << '.';
  throw std::runtime_error(ss.str());
}

}  // namespace primitiv
