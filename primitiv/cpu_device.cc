#include <config.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <primitiv/cpu_device.h>
#include <primitiv/error.h>

using std::cerr;
using std::endl;

namespace primitiv {

CPUDevice::CPUDevice() : blocks_(), rng_(std::random_device()()) {}
CPUDevice::CPUDevice(unsigned rng_seed) : blocks_(), rng_(rng_seed) {}

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

Tensor CPUDevice::new_tensor_impl(const Shape &shape) {
  const unsigned mem_size = sizeof(float) * shape.size();
  void *data = std::malloc(mem_size);
  if (!data) {
    THROW_ERROR("Memory allocation failed. Requested size: " << mem_size);
  }
  blocks_.insert(std::make_pair(data, mem_size));
  return Tensor(shape, this, data);
}

void CPUDevice::delete_tensor_impl(Tensor &x) {
  void *data = x.data();
  auto it = blocks_.find(data);
  if (it == blocks_.end()) {
    THROW_ERROR("Attempted to dispose unknown memory block: " << data);
  }
  blocks_.erase(it);
  std::free(data);
}

#define DATA(x) static_cast<float *>((x).data())
#define CDATA(x) static_cast<const float *>((x).data())

#define REPEAT_OP(i, n, op) \
  for (unsigned (i) = 0; (i) < (n); ++(i)) { (op); }

std::vector<float> CPUDevice::tensor_to_vector_impl(const Tensor &x) {
  const unsigned num_elements = x.shape().size();
  std::vector<float> ret(num_elements);
  std::memcpy(&ret[0], x.data(), sizeof(float) * num_elements);
  return ret;
}

void CPUDevice::reset_tensor_impl(Tensor &x, float k) {
  float *dest = DATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = k);
}

void CPUDevice::reset_tensor_impl(Tensor &x, const std::vector<float> &values) {
  std::memcpy(x.data(), &values[0], sizeof(float) * x.shape().size());
}

Tensor CPUDevice::random_bernoulli_impl(const Shape &shape, float p) {
  std::bernoulli_distribution dist(p);
  Tensor ret = new_tensor(shape);
  float *dest = DATA(ret);
  const unsigned size = shape.size();
  for (unsigned i = 0; i < size; ++i) {
    dest[i] = dist(rng_);
  }
  return ret;
}

Tensor CPUDevice::random_uniform_impl(
    const Shape &shape, float lower, float upper) {
  std::uniform_real_distribution<float> dist(lower, upper);
  Tensor ret = new_tensor(shape);
  float *dest = DATA(ret);
  const unsigned size = shape.size();
  for (unsigned i = 0; i < size; ++i) {
    const float x = dist(rng_);
    dest[i] = x == lower ? upper : x;
  }
  return ret;
}

Tensor CPUDevice::random_normal_impl(const Shape &shape, float mean, float sd) {
  std::normal_distribution<float> dist(mean, sd);
  Tensor ret = new_tensor(shape);
  float *dest = DATA(ret);
  const unsigned size = shape.size();
  for (unsigned i = 0; i < size; ++i) {
    dest[i] = dist(rng_);
  }
  return ret;
}

Tensor CPUDevice::slice_impl(
    const Tensor &x, unsigned dim, unsigned lower, unsigned upper) {
  const Shape &s = x.shape();
  std::vector<unsigned> dims = s.dims();
  const unsigned bs = s.batch_size();
  const unsigned diff = upper - lower;
  unsigned base = 1;
  for (unsigned i = 0; i < dim; ++i) base *= dims[i];
  const unsigned offset = base * lower;
  const unsigned span = base * diff;
  const unsigned skip = base * (dims[dim] - diff);
  unsigned repeat = bs;
  for (unsigned i = dim + 1; i < dims.size(); ++i) repeat *= dims[i];
  dims[dim] = diff;

  Tensor ret = new_tensor(Shape(dims, bs));
  float *dest = DATA(ret);
  const float *src = CDATA(x) + offset;
  for (unsigned i = 0; i < repeat; ++i) {
    for (unsigned j = 0; j < span; ++j) {
      *dest++ = *src++;
    }
    src += skip;
  }
  return ret;
}

Tensor CPUDevice::concat_impl(
    const std::vector<const Tensor *> &xs,
    unsigned dim, const Shape &new_shape) {
  const std::vector<unsigned> new_dims = new_shape.dims();
  const unsigned new_bs = new_shape.batch_size();
  unsigned base = 1;
  for (unsigned i = 0; i < dim; ++i) base *= new_dims[i];
  unsigned repeat = 1;
  for (unsigned i = dim + 1; i < new_dims.size(); ++i) repeat *= new_dims[i];

  Tensor ret = new_tensor(new_shape);
  unsigned offset = 0;
  for (const Tensor *x : xs) {
    const unsigned src_dim = x->shape().dim(dim);
    const unsigned span = base * src_dim;
    const unsigned skip = base * (new_dims[dim] - src_dim);
    const unsigned b_skip = (x->shape().batch_size() > 1) * span * repeat;
    float *dest = DATA(ret) + offset;
    for (unsigned b = 0; b < new_bs; ++b) {
      const float *src = CDATA(*x) + b * b_skip;
      for (unsigned i = 0; i < repeat; ++i) {
        for (unsigned j = 0; j < span; ++j) {
          *dest++ = *src++;
        }
        dest += skip;
      }
    }
    offset += span;
  }
  return ret;
}

Tensor CPUDevice::duplicate_impl(const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  std::memcpy(ret.data(), x.data(), sizeof(float) * x.shape().size());
  return ret;
}

Tensor CPUDevice::negate_impl(const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = -src[i]);
  return ret;
}

Tensor CPUDevice::add_impl(const Tensor &x, float k) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = src[i] + k);
  return ret;
}

Tensor CPUDevice::add_impl(const Tensor &a, const Tensor &b) {
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const float *src_a = CDATA(a);
  const float *src_b = CDATA(b);

  if (sa.batch_size() == sb.batch_size()) {
    // ret = a + b
    Tensor ret = new_tensor(sa);
    float *dest = DATA(ret);
    const unsigned size = sa.size();
    REPEAT_OP(i, size, dest[i] = src_a[i] + src_b[i]);
    return ret;
  } else if (sa.batch_size() == 1) {
    // ret = batch_broadcast(a) + b
    Tensor ret = new_tensor(sb);
    float *dest = DATA(ret);
    const unsigned ms = sa.size();
    const unsigned bs = sb.batch_size();
    for (unsigned k = 0; k < bs; ++k, dest += ms, src_b += ms) {
      REPEAT_OP(i, ms, dest[i] = src_a[i] + src_b[i]);
    }
    return ret;
  } else /* sb.batch_size() == 1 */ {
    // ret = a + batch_broadcast(b)
    Tensor ret = new_tensor(sa);
    float *dest = DATA(ret);
    const unsigned ms = sb.size();
    const unsigned bs = sa.batch_size();
    for (unsigned k = 0; k < bs; ++k, dest += ms, src_a += ms) {
      REPEAT_OP(i, ms, dest[i] = src_a[i] + src_b[i]);
    }
    return ret;
  }
}

Tensor CPUDevice::subtract_impl(const Tensor &x, float k) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = src[i] - k);
  return ret;
}

Tensor CPUDevice::subtract_impl(float k, const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = k - src[i]);
  return ret;
}

Tensor CPUDevice::subtract_impl(const Tensor &a, const Tensor &b) {
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const float *src_a = CDATA(a);
  const float *src_b = CDATA(b);

  if (sa.batch_size() == sb.batch_size()) {
    // ret = a - b
    Tensor ret = new_tensor(sa);
    float *dest = DATA(ret);
    const unsigned size = sa.size();
    REPEAT_OP(i, size, dest[i] = src_a[i] - src_b[i]);
    return ret;
  } else if (sa.batch_size() == 1) {
    // ret = batch_broadcast(a) - b
    Tensor ret = new_tensor(sb);
    float *dest = DATA(ret);
    const unsigned ms = sa.size();
    const unsigned bs = sb.batch_size();
    for (unsigned k = 0; k < bs; ++k, dest += ms, src_b += ms) {
      REPEAT_OP(i, ms, dest[i] = src_a[i] - src_b[i]);
    }
    return ret;
  } else /* sb.batch_size() == 1 */ {
    // ret = a - batch_broadcast(b)
    Tensor ret = new_tensor(sa);
    float *dest = DATA(ret);
    const unsigned ms = sb.size();
    const unsigned bs = sa.batch_size();
    for (unsigned k = 0; k < bs; ++k, dest += ms, src_a += ms) {
      REPEAT_OP(i, ms, dest[i] = src_a[i] - src_b[i]);
    }
    return ret;
  }
}

Tensor CPUDevice::multiply_impl(const Tensor &x, float k) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = src[i] * k);
  return ret;
}

Tensor CPUDevice::multiply_impl(const Tensor &a, const Tensor &b) {
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const float *src_a = CDATA(a);
  const float *src_b = CDATA(b);

  if (sa.batch_size() == sb.batch_size()) {
    // ret = a * b
    Tensor ret = new_tensor(sa);
    float *dest = DATA(ret);
    const unsigned size = sa.size();
    REPEAT_OP(i, size, dest[i] = src_a[i] * src_b[i]);
    return ret;
  } else if (sa.batch_size() == 1) {
    // ret = batch_broadcast(a) * b
    Tensor ret = new_tensor(sb);
    float *dest = DATA(ret);
    const unsigned ms = sa.size();
    const unsigned bs = sb.batch_size();
    for (unsigned k = 0; k < bs; ++k, dest += ms, src_b += ms) {
      REPEAT_OP(i, ms, dest[i] = src_a[i] * src_b[i]);
    }
    return ret;
  } else /* sb.batch_size() == 1 */ {
    // ret = a * batch_broadcast(b)
    Tensor ret = new_tensor(sa);
    float *dest = DATA(ret);
    const unsigned ms = sb.size();
    const unsigned bs = sa.batch_size();
    for (unsigned k = 0; k < bs; ++k, dest += ms, src_a += ms) {
      REPEAT_OP(i, ms, dest[i] = src_a[i] * src_b[i]);
    }
    return ret;
  }
}

Tensor CPUDevice::divide_impl(const Tensor &x, float k) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = src[i] / k);
  return ret;
}

Tensor CPUDevice::divide_impl(float k, const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = k / src[i]);
  return ret;
}

Tensor CPUDevice::divide_impl(const Tensor &a, const Tensor &b) {
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const float *src_a = CDATA(a);
  const float *src_b = CDATA(b);

  if (sa.batch_size() == sb.batch_size()) {
    // ret = a / b
    Tensor ret = new_tensor(sa);
    float *dest = DATA(ret);
    const unsigned size = sa.size();
    REPEAT_OP(i, size, dest[i] = src_a[i] / src_b[i]);
    return ret;
  } else if (sa.batch_size() == 1) {
    // ret = batch_broadcast(a) / b
    Tensor ret = new_tensor(sb);
    float *dest = DATA(ret);
    const unsigned ms = sa.size();
    const unsigned bs = sb.batch_size();
    for (unsigned k = 0; k < bs; ++k, dest += ms, src_b += ms) {
      REPEAT_OP(i, ms, dest[i] = src_a[i] / src_b[i]);
    }
    return ret;
  } else /* sb.batch_size() == 1 */ {
    // ret = a / batch_broadcast(b)
    Tensor ret = new_tensor(sa);
    float *dest = DATA(ret);
    const unsigned ms = sb.size();
    const unsigned bs = sa.batch_size();
    for (unsigned k = 0; k < bs; ++k, dest += ms, src_a += ms) {
      REPEAT_OP(i, ms, dest[i] = src_a[i] / src_b[i]);
    }
    return ret;
  }
}

Tensor CPUDevice::transpose_impl(const Tensor &x) {
  const Shape &s = x.shape();
  const unsigned d1 = s.dim(0);
  const unsigned d2 = s.dim(1);
  const unsigned ms = d1 * d2;
  const unsigned bs = s.batch_size();
  Tensor ret = new_tensor(Shape({d2, d1}, bs));
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

Tensor CPUDevice::dot_impl(const Tensor &a, const Tensor &b) {
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

  if (sa.batch_size() == sb.batch_size()) {
    // ret = a . b
    const unsigned bs = sa.batch_size();
    Tensor ret = new_tensor(Shape({d1, d3}, bs));
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
    Tensor ret = new_tensor(Shape({d1, d3}, bs));
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
  } else /* sb.batch_size() == 1 */ {
    // ret = a . batch_broadcast(b)
    const unsigned bs = sa.batch_size();
    Tensor ret = new_tensor(Shape({d1, d3}, bs));
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

Tensor CPUDevice::exp_impl(const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = std::exp(src[i]));
  return ret;
}

Tensor CPUDevice::tanh_impl(const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = std::tanh(src[i]));
  return ret;
}

Tensor CPUDevice::sigmoid_impl(const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = .5 + .5 * std::tanh(.5 * src[i]));
  return ret;
}

Tensor CPUDevice::step_impl(const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = static_cast<float>(src[i] > 0));
  return ret;
}

Tensor CPUDevice::relu_impl(const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = std::max(src[i], .0f));
  return ret;
}

Tensor CPUDevice::batch_sum_impl(const Tensor &x) {
  Tensor ret = new_tensor(Shape(x.shape().dims()));
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned bs = x.shape().batch_size();
  const unsigned size = ret.shape().size();
  for (unsigned i = 0; i < size; ++i) {
    float temp = 0;
    for (unsigned b = 0, pos = i; b < bs; ++b, pos += size) {
      temp += src[pos];
    }
    dest[i] = temp;
  }
  return ret;
}

void CPUDevice::add_gradient_impl(Tensor &a, const Tensor &b) {
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  float *dest = DATA(a);
  const float *src = CDATA(b);

  if (sa.batch_size() == sb.batch_size()) {
    // a += b
    const unsigned size = sa.size();
    REPEAT_OP(i, size, dest[i] += src[i]);
  } else if (sa.batch_size() == 1) {
    // a += batch_sum(b)
    const unsigned ms = sa.size();
    const unsigned bs = sb.batch_size();
    for (unsigned k = 0; k < bs; ++k, src += ms) {
      REPEAT_OP(i, ms, dest[i] += src[i]);
    }
  } else if (sb.batch_size() == 1) {
    // a += batch_broadcast(b)
    const unsigned ms = sb.size();
    const unsigned bs = sa.batch_size();
    for (unsigned k = 0; k < bs; ++k, dest += ms) {
      REPEAT_OP(i, ms, dest[i] += src[i]);
    }
  }
}

}  // namespace primitiv
