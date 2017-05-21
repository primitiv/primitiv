#include <config.h>

#include <cstdlib>
#include <cstring>
#include <cmath>
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
  if ((x).device() != this) { \
    std::stringstream ss; \
    ss << "Device mismatched. (" #x ").device(): " << (x).device() \
       << "!= this:" << this; \
    throw std::runtime_error(ss.str()); \
  }

#define DATA(x) static_cast<float *>((x).data());
#define CDATA(x) static_cast<const float *>((x).data());

#define REPEAT_OP(i, n, op) \
  for (unsigned (i) = 0; (i) < (n); ++(i)) { \
    (op); \
  }

Tensor CPUDevice::constant(const Shape &shape, const float k) {
  Tensor ret(shape, this);
  float *dest = DATA(ret);
  const unsigned size = shape.size();
  REPEAT_OP(i, size, dest[i] = k);
  return ret;
}

Tensor CPUDevice::duplicate(const Tensor &x) {
  CHECK_DEVICE(x);

  Tensor ret(x.shape(), this);
  std::memcpy(ret.data(), x.data(), sizeof(float) * x.shape().size());
  return ret;
}

Tensor CPUDevice::negate(const Tensor &x) {
  CHECK_DEVICE(x);

  Tensor ret(x.shape(), this);
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = -src[i]);
  return ret;
}

Tensor CPUDevice::add(const Tensor &x, const float k) {
  CHECK_DEVICE(x);

  Tensor ret(x.shape(), this);
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = src[i] + k);
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
      Tensor ret(sa, this);
      float *dest = DATA(ret);
      const unsigned size = sa.size();
      REPEAT_OP(i, size, dest[i] = src_a[i] + src_b[i]);
      return ret;
    } else if (sa.batch_size() == 1) {
      // ret = batch_broadcast(a) + b
      Tensor ret(sb, this);
      float *dest = DATA(ret);
      const unsigned ms = sa.size();
      const unsigned bs = sb.batch_size();
      for (unsigned k = 0; k < bs; ++k, dest += ms, src_b += ms) {
        REPEAT_OP(i, ms, dest[i] = src_a[i] + src_b[i]);
      }
      return ret;
    } else if (sb.batch_size() == 1) {
      // ret = a + batch_broadcast(b)
      Tensor ret(sa, this);
      float *dest = DATA(ret);
      const unsigned ms = sb.size();
      const unsigned bs = sa.batch_size();
      for (unsigned k = 0; k < bs; ++k, dest += ms, src_a += ms) {
        REPEAT_OP(i, ms, dest[i] = src_a[i] + src_b[i]);
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

  Tensor ret(x.shape(), this);
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = src[i] - k);
  return ret;
}

Tensor CPUDevice::subtract(const float k, const Tensor &x) {
  CHECK_DEVICE(x);

  Tensor ret(x.shape(), this);
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = k - src[i]);
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
      Tensor ret(sa, this);
      float *dest = DATA(ret);
      const unsigned size = sa.size();
      REPEAT_OP(i, size, dest[i] = src_a[i] - src_b[i]);
      return ret;
    } else if (sa.batch_size() == 1) {
      // ret = batch_broadcast(a) - b
      Tensor ret(sb, this);
      float *dest = DATA(ret);
      const unsigned ms = sa.size();
      const unsigned bs = sb.batch_size();
      for (unsigned k = 0; k < bs; ++k, dest += ms, src_b += ms) {
        REPEAT_OP(i, ms, dest[i] = src_a[i] - src_b[i]);
      }
      return ret;
    } else if (sb.batch_size() == 1) {
      // ret = a - batch_broadcast(b)
      Tensor ret(sa, this);
      float *dest = DATA(ret);
      const unsigned ms = sb.size();
      const unsigned bs = sa.batch_size();
      for (unsigned k = 0; k < bs; ++k, dest += ms, src_a += ms) {
        REPEAT_OP(i, ms, dest[i] = src_a[i] - src_b[i]);
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

  Tensor ret(x.shape(), this);
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = src[i] * k);
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
      Tensor ret(sa, this);
      float *dest = DATA(ret);
      const unsigned size = sa.size();
      REPEAT_OP(i, size, dest[i] = src_a[i] * src_b[i]);
      return ret;
    } else if (sa.batch_size() == 1) {
      // ret = batch_broadcast(a) * b
      Tensor ret(sb, this);
      float *dest = DATA(ret);
      const unsigned ms = sa.size();
      const unsigned bs = sb.batch_size();
      for (unsigned k = 0; k < bs; ++k, dest += ms, src_b += ms) {
        REPEAT_OP(i, ms, dest[i] = src_a[i] * src_b[i]);
      }
      return ret;
    } else if (sb.batch_size() == 1) {
      // ret = a * batch_broadcast(b)
      Tensor ret(sa, this);
      float *dest = DATA(ret);
      const unsigned ms = sb.size();
      const unsigned bs = sa.batch_size();
      for (unsigned k = 0; k < bs; ++k, dest += ms, src_a += ms) {
        REPEAT_OP(i, ms, dest[i] = src_a[i] * src_b[i]);
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

  Tensor ret(x.shape(), this);
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = src[i] / k);
  return ret;
}

Tensor CPUDevice::divide(const float k, const Tensor &x) {
  CHECK_DEVICE(x);

  Tensor ret(x.shape(), this);
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = k / src[i]);
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
      Tensor ret(sa, this);
      float *dest = DATA(ret);
      const unsigned size = sa.size();
      REPEAT_OP(i, size, dest[i] = src_a[i] / src_b[i]);
      return ret;
    } else if (sa.batch_size() == 1) {
      // ret = batch_broadcast(a) / b
      Tensor ret(sb, this);
      float *dest = DATA(ret);
      const unsigned ms = sa.size();
      const unsigned bs = sb.batch_size();
      for (unsigned k = 0; k < bs; ++k, dest += ms, src_b += ms) {
        REPEAT_OP(i, ms, dest[i] = src_a[i] / src_b[i]);
      }
      return ret;
    } else if (sb.batch_size() == 1) {
      // ret = a / batch_broadcast(b)
      Tensor ret(sa, this);
      float *dest = DATA(ret);
      const unsigned ms = sb.size();
      const unsigned bs = sa.batch_size();
      for (unsigned k = 0; k < bs; ++k, dest += ms, src_a += ms) {
        REPEAT_OP(i, ms, dest[i] = src_a[i] / src_b[i]);
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

Tensor CPUDevice::transpose(const Tensor &x) {
  CHECK_DEVICE(x);
  const Shape &s = x.shape();
  if (s.dims().size() > 2) {
    std::stringstream ss;
    ss << "Attempted to transpose a tensor with shape "
       << x.shape().to_string() << '.';
    throw std::runtime_error(ss.str());
  }

  const unsigned d1 = s.dim(0);
  const unsigned d2 = s.dim(1);
  const unsigned ms = d1 * d2;
  const unsigned bs = s.batch_size();
  Tensor ret(Shape({d2, d1}, bs), this);
  float *dest = DATA(ret);
  const float *src = CDATA(x);

  for (unsigned k = 0; k < bs; ++k) {
    float *pd = dest;
    const float *ps = src;
    for (unsigned j = 0; j < d2; ++j) {
      float *ppd = pd;
      for (unsigned i = 0; i < d1; ++i) {
        *ppd = ps[i];
        ppd += d2;
      }
      ++pd;
      ps += d1;
    }
    dest += ms;
    src += ms;
  }

  return ret;
}

Tensor CPUDevice::dot(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const float *src_a = CDATA(a);
  const float *src_b = CDATA(b);
  const unsigned d1 = sa.dim(0);
  const unsigned d2 = sa.dim(1);
  const unsigned d3 = sb.dim(1);
  const unsigned dest_shift = d1 * d3;
  const unsigned src_a_shift = d1 * d2;
  const unsigned src_b_shift = d2 * d3;

  if (sa.dims().size() <= 2 && sb.dims().size() <= 2 && d2 == sb.dim(0)) {
    if (sa.batch_size() == sb.batch_size()) {
      // ret = a . b
      const unsigned bs = sa.batch_size();
      Tensor ret(Shape({d1, d3}, bs), this);
      float *dest = DATA(ret);
      for (unsigned b = 0; b < bs; ++b) {
        for (unsigned i = 0; i < d1; ++i) {
          for (unsigned k = 0; k < d3; ++k) {
            float tmp = 0;
            for (unsigned j = 0; j < d2; ++j) {
              tmp += src_a[i + j * d1] * src_b[j + k * d2];
            }
            dest[i + k * d1] = tmp;
          }
        }
        dest += dest_shift;
        src_a += src_a_shift;
        src_b += src_b_shift;
      }
      return ret;
    } else if (sa.batch_size() == 1) {
      // ret = batch_broadcast(a) . b
      const unsigned bs = sb.batch_size();
      Tensor ret(Shape({d1, d3}, bs), this);
      float *dest = DATA(ret);
      for (unsigned b = 0; b < bs; ++b) {
        for (unsigned i = 0; i < d1; ++i) {
          for (unsigned k = 0; k < d3; ++k) {
            float tmp = 0;
            for (unsigned j = 0; j < d2; ++j) {
              tmp += src_a[i + j * d1] * src_b[j + k * d2];
            }
            dest[i + k * d1] = tmp;
          }
        }
        dest += dest_shift;
        src_b += src_b_shift;
      }
      return ret;
    } else if (sb.batch_size() == 1) {
      // ret = a . batch_broadcast(b)
      const unsigned bs = sa.batch_size();
      Tensor ret(Shape({d1, d3}, bs), this);
      float *dest = DATA(ret);
      for (unsigned b = 0; b < bs; ++b) {
        for (unsigned i = 0; i < d1; ++i) {
          for (unsigned k = 0; k < d3; ++k) {
            float tmp = 0;
            for (unsigned j = 0; j < d2; ++j) {
              tmp += src_a[i + j * d1] * src_b[j + k * d2];
            }
            dest[i + k * d1] = tmp;
          }
        }
        dest += dest_shift;
        src_a += src_a_shift;
      }
      return ret;
    }
  }

  // error
  std::stringstream ss;
  ss << "Attempted to calculate the dot product of tensors with shapes "
     << a.shape().to_string() << " and " << b.shape().to_string() << '.';
  throw std::runtime_error(ss.str());
}

Tensor CPUDevice::exp(const Tensor &x) {
  CHECK_DEVICE(x);

  Tensor ret(x.shape(), this);
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = std::exp(src[i]));
  return ret;
}

Tensor CPUDevice::tanh(const Tensor &x) {
  CHECK_DEVICE(x);

  Tensor ret(x.shape(), this);
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = std::tanh(src[i]));
  return ret;
}

void CPUDevice::add_gradient(Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  float *dest = DATA(a);
  const float *src = CDATA(b);

  if (sa.dims() == sb.dims()) {
    if (sa.batch_size() == sb.batch_size()) {
      // a += b
      const unsigned size = sa.size();
      REPEAT_OP(i, size, dest[i] += src[i]);
      return;
    } else if (sa.batch_size() == 1) {
      // a += batch_sum(b)
      const unsigned ms = sa.size();
      const unsigned bs = sb.batch_size();
      for (unsigned k = 0; k < bs; ++k, src += ms) {
        REPEAT_OP(i, ms, dest[i] += src[i]);
      }
      return;
    } else if (sb.batch_size() == 1) {
      // a += batch_broadcast(b)
      const unsigned ms = sb.size();
      const unsigned bs = sa.batch_size();
      for (unsigned k = 0; k < bs; ++k, dest += ms) {
        REPEAT_OP(i, ms, dest[i] += src[i]);
      }
      return;
    }
  }

  // error
  std::stringstream ss;
  ss << "Attempted to add gradient tensors with shapes "
     << a.shape().to_string() << " and " << b.shape().to_string() << '.';
  throw std::runtime_error(ss.str());
}

}  // namespace primitiv
