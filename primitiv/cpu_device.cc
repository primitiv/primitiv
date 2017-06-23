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

std::shared_ptr<void> CPUDevice::new_handle(const Shape &shape) {
  const unsigned mem_size = sizeof(float) * shape.num_total_elements();
  void *data = std::malloc(mem_size);
  if (!data) {
    THROW_ERROR("Memory allocation failed. Requested size: " << mem_size);
  }
  return std::shared_ptr<void>(data, std::free);
}

#define DATA(x) static_cast<float *>((x).data())
#define CDATA(x) static_cast<const float *>((x).data())

#define REPEAT_OP(i, n, op) \
  for (unsigned (i) = 0; (i) < (n); ++(i)) { (op); }

std::vector<float> CPUDevice::tensor_to_vector_impl(const Tensor &x) {
  const unsigned num_elements = x.shape().num_total_elements();
  std::vector<float> ret(num_elements);
  std::memcpy(&ret[0], x.data(), sizeof(float) * num_elements);
  return ret;
}

void CPUDevice::reset_tensor_impl(Tensor &x, float k) {
  float *dest = DATA(x);
  const unsigned size = x.shape().num_total_elements();
  REPEAT_OP(i, size, dest[i] = k);
}

void CPUDevice::reset_tensor_by_array_impl(Tensor &x, const float values[]) {
  std::memcpy(x.data(), values, sizeof(float) * x.shape().num_total_elements());
}

Tensor CPUDevice::copy_tensor_impl(const Tensor &x) {
  switch (x.device()->type()) {
    case Device::DEVICE_TYPE_CPU:
      return new_tensor_by_array(
          x.shape(), reinterpret_cast<const float *>(x.data()));
    default:
      return new_tensor_by_vector(x.shape(), x.to_vector());
  }
}

Tensor CPUDevice::random_bernoulli_impl(const Shape &shape, float p) {
  std::bernoulli_distribution dist(p);
  Tensor ret = new_tensor(shape);
  float *dest = DATA(ret);
  const unsigned size = shape.num_total_elements();
  REPEAT_OP(i, size, dest[i] = dist(rng_));
  return ret;
}

Tensor CPUDevice::random_uniform_impl(
    const Shape &shape, float lower, float upper) {
  std::uniform_real_distribution<float> dist(lower, upper);
  Tensor ret = new_tensor(shape);
  float *dest = DATA(ret);
  const unsigned size = shape.num_total_elements();
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
  const unsigned size = shape.num_total_elements();
  REPEAT_OP(i, size, dest[i] = dist(rng_));
  return ret;
}

Tensor CPUDevice::pick_impl(
    const Tensor &x, unsigned dim,
    const std::vector<unsigned> &ids, Shape &&new_shape) {
  const unsigned bs = new_shape.batch_size();
  const unsigned skip_x =
    (x.shape().batch_size() > 1) * x.shape().num_elements_per_sample();
  const unsigned skip_i = ids.size() > 1;
  const unsigned base = new_shape.num_elements_under_rank(dim);
  const unsigned skip = base * x.shape()[dim];
  const unsigned repeat = new_shape.num_elements_per_sample() / base;

  Tensor ret = new_tensor(new_shape);
  float *dest = DATA(ret);
  for (unsigned batch = 0; batch < bs; ++batch) {
    const float *src = CDATA(x) + batch * skip_x + base * ids[batch * skip_i];
    for (unsigned i = 0; i < repeat; ++i) {
      const float *sp = src;
      REPEAT_OP(j, base, *dest++ = *sp++);
      src += skip;
    }
  }
  return ret;
}

Tensor CPUDevice::slice_impl(
    const Tensor &x, unsigned dim, unsigned offset, Shape &&new_shape) {
  const unsigned base = new_shape.num_elements_under_rank(dim);
  const unsigned span = base * new_shape[dim];
  const unsigned skip = base * x.shape()[dim];
  const unsigned repeat = new_shape.num_total_elements() / span;

  Tensor ret = new_tensor(new_shape);
  float *dest = DATA(ret);
  const float *src = CDATA(x) + base * offset;
  for (unsigned i = 0; i < repeat; ++i) {
    const float *sp = src;
    REPEAT_OP(j, span, *dest++ = *sp++);
    src += skip;
  }
  return ret;
}

Tensor CPUDevice::concat_impl(
    const std::vector<const Tensor *> &xs, unsigned dim, Shape &&new_shape) {
  const unsigned new_bs = new_shape.batch_size();
  const unsigned base = new_shape.num_elements_under_rank(dim);
  const unsigned skip = base * new_shape[dim];
  const unsigned repeat = new_shape.num_elements_per_sample() / skip;

  Tensor ret = new_tensor(new_shape);
  unsigned offset = 0;
  for (const Tensor *x : xs) {
    const unsigned src_dim = x->shape()[dim];
    const unsigned span = base * src_dim;
    const unsigned b_skip = (x->shape().batch_size() > 1) * span * repeat;
    float *dest = DATA(ret) + offset;
    const float *src = CDATA(*x);
    for (unsigned batch = 0; batch < new_bs; ++batch) {
      const float *sp = src;
      for (unsigned i = 0; i < repeat; ++i) {
        float *dp = dest;
        REPEAT_OP(j, span, *dp++ = *sp++);
        dest += skip;
      }
      src += b_skip;
    }
    offset += span;
  }
  return ret;
}

Tensor CPUDevice::duplicate_impl(const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  std::memcpy(ret.data(), x.data(), sizeof(float) * x.shape().num_total_elements());
  return ret;
}

Tensor CPUDevice::negate_impl(const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().num_total_elements();
  REPEAT_OP(i, size, dest[i] = -src[i]);
  return ret;
}

Tensor CPUDevice::add_impl(const Tensor &x, float k) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().num_total_elements();
  REPEAT_OP(i, size, dest[i] = src[i] + k);
  return ret;
}

Tensor CPUDevice::add_impl(
    const Tensor &a, const Tensor &b, Shape &&new_shape) {
  const unsigned size = new_shape.num_elements_per_sample();
  const unsigned bs = new_shape.batch_size();
  const unsigned skip_a = (a.shape().batch_size() > 1) * size;
  const unsigned skip_b = (b.shape().batch_size() > 1) * size;
  Tensor ret = new_tensor(new_shape);
  float *dest = DATA(ret);
  const float *src_a = CDATA(a);
  const float *src_b = CDATA(b);
  for (unsigned batch = 0; batch < bs; ++batch) {
    REPEAT_OP(i, size, dest[i] = src_a[i] + src_b[i]);
    dest += size;
    src_a += skip_a;
    src_b += skip_b;
  }
  return ret;
}

Tensor CPUDevice::subtract_impl(const Tensor &x, float k) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().num_total_elements();
  REPEAT_OP(i, size, dest[i] = src[i] - k);
  return ret;
}

Tensor CPUDevice::subtract_impl(float k, const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().num_total_elements();
  REPEAT_OP(i, size, dest[i] = k - src[i]);
  return ret;
}

Tensor CPUDevice::subtract_impl(
    const Tensor &a, const Tensor &b, Shape &&new_shape) {
  const unsigned size = new_shape.num_elements_per_sample();
  const unsigned bs = new_shape.batch_size();
  const unsigned skip_a = (a.shape().batch_size() > 1) * size;
  const unsigned skip_b = (b.shape().batch_size() > 1) * size;
  Tensor ret = new_tensor(new_shape);
  float *dest = DATA(ret);
  const float *src_a = CDATA(a);
  const float *src_b = CDATA(b);
  for (unsigned batch = 0; batch < bs; ++batch) {
    REPEAT_OP(i, size, dest[i] = src_a[i] - src_b[i]);
    dest += size;
    src_a += skip_a;
    src_b += skip_b;
  }
  return ret;
}

Tensor CPUDevice::multiply_impl(const Tensor &x, float k) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().num_total_elements();
  REPEAT_OP(i, size, dest[i] = src[i] * k);
  return ret;
}

Tensor CPUDevice::multiply_impl(
    const Tensor &a, const Tensor &b, Shape &&new_shape) {
  const unsigned size = new_shape.num_elements_per_sample();
  const unsigned bs = new_shape.batch_size();
  const unsigned skip_a = (a.shape().batch_size() > 1) * size;
  const unsigned skip_b = (b.shape().batch_size() > 1) * size;
  Tensor ret = new_tensor(new_shape);
  float *dest = DATA(ret);
  const float *src_a = CDATA(a);
  const float *src_b = CDATA(b);
  for (unsigned batch = 0; batch < bs; ++batch) {
    REPEAT_OP(i, size, dest[i] = src_a[i] * src_b[i]);
    dest += size;
    src_a += skip_a;
    src_b += skip_b;
  }
  return ret;
}

Tensor CPUDevice::divide_impl(const Tensor &x, float k) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().num_total_elements();
  REPEAT_OP(i, size, dest[i] = src[i] / k);
  return ret;
}

Tensor CPUDevice::divide_impl(float k, const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().num_total_elements();
  REPEAT_OP(i, size, dest[i] = k / src[i]);
  return ret;
}

Tensor CPUDevice::divide_impl(
    const Tensor &a, const Tensor &b, Shape &&new_shape) {
  const unsigned size = new_shape.num_elements_per_sample();
  const unsigned bs = new_shape.batch_size();
  const unsigned skip_a = (a.shape().batch_size() > 1) * size;
  const unsigned skip_b = (b.shape().batch_size() > 1) * size;
  Tensor ret = new_tensor(new_shape);
  float *dest = DATA(ret);
  const float *src_a = CDATA(a);
  const float *src_b = CDATA(b);
  for (unsigned batch = 0; batch < bs; ++batch) {
    REPEAT_OP(i, size, dest[i] = src_a[i] / src_b[i]);
    dest += size;
    src_a += skip_a;
    src_b += skip_b;
  }
  return ret;
}

Tensor CPUDevice::transpose_impl(const Tensor &x, Shape &&new_shape) {
  const unsigned d1 = new_shape[1];
  const unsigned d2 = new_shape[0];
  const unsigned ms = d1 * d2;
  const unsigned bs = new_shape.batch_size();
  Tensor ret = new_tensor(new_shape);
  float *dest = DATA(ret);
  const float *src = CDATA(x);

  for (unsigned k = 0; k < bs; ++k) {
    float *pd = dest;
    for (unsigned j = 0; j < d2; ++j) {
      float *ppd = pd;
      for (unsigned i = 0; i < d1; ++i) {
        *ppd = *src++;
        ppd += d2;
      }
      ++pd;
    }
    dest += ms;
  }

  return ret;
}

Tensor CPUDevice::dot_impl(
    const Tensor &a, const Tensor &b, Shape &&new_shape) {
  const unsigned d1 = new_shape[0];
  const unsigned d2 = a.shape()[1];
  const unsigned d3 = new_shape[1];
  const unsigned bs = new_shape.batch_size();
  const unsigned dest_shift = d1 * d3;
  const unsigned src_a_shift = (a.shape().batch_size() > 1) * d1 * d2;
  const unsigned src_b_shift = (b.shape().batch_size() > 1) * d2 * d3;

  Tensor ret = new_tensor(new_shape);
  float *dest = DATA(ret);
  const float *src_a = CDATA(a);
  const float *src_b = CDATA(b);
  for (unsigned batch = 0; batch < bs; ++batch) {
    for (unsigned i = 0; i < d1; ++i) {
      for (unsigned ky = 0, kb = 0; ky < dest_shift; ky += d1, kb += d2) {
        float tmp = 0;
        for (unsigned ja = 0, jb = 0; jb < d2; ja += d1, ++jb) {
          tmp += src_a[i + ja] * src_b[jb + kb];
        }
        dest[i + ky] = tmp;
      }
    }
    dest += dest_shift;
    src_a += src_a_shift;
    src_b += src_b_shift;
  }
  return ret;
}

Tensor CPUDevice::exp_impl(const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().num_total_elements();
  REPEAT_OP(i, size, dest[i] = std::exp(src[i]));
  return ret;
}

Tensor CPUDevice::tanh_impl(const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().num_total_elements();
  REPEAT_OP(i, size, dest[i] = std::tanh(src[i]));
  return ret;
}

Tensor CPUDevice::sigmoid_impl(const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().num_total_elements();
  REPEAT_OP(i, size, dest[i] = .5 + .5 * std::tanh(.5 * src[i]));
  return ret;
}

Tensor CPUDevice::step_impl(const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().num_total_elements();
  REPEAT_OP(i, size, dest[i] = static_cast<float>(src[i] > 0));
  return ret;
}

Tensor CPUDevice::relu_impl(const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned size = x.shape().num_total_elements();
  REPEAT_OP(i, size, dest[i] = std::max(src[i], .0f));
  return ret;
}

Tensor CPUDevice::sum_impl(const Tensor &x, unsigned dim) {
  const Shape new_shape = x.shape().resize_dim(dim, 1);
  const unsigned n = x.shape()[dim];
  const unsigned repeat = new_shape.num_total_elements();
  const unsigned skip1 = new_shape.num_elements_under_rank(dim);
  const unsigned skip2 = skip1 * n;
  Tensor ret = new_tensor(new_shape);
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  for (unsigned i = 0; i < repeat; ++i) {
    unsigned offset = i % skip1 + (i / skip1) * skip2;
    float tmp = 0;
    for (unsigned j = 0; j < n; ++j) {
      tmp += src[offset];
      offset += skip1;
    }
    dest[i] = tmp;
  }
  return ret;
}

Tensor CPUDevice::logsumexp_impl(const Tensor &x, unsigned dim) {
  const Shape new_shape = x.shape().resize_dim(dim, 1);
  const unsigned n = x.shape()[dim];
  const unsigned repeat = new_shape.num_total_elements();
  const unsigned skip1 = new_shape.num_elements_under_rank(dim);
  const unsigned skip2 = skip1 * n;
  Tensor ret = new_tensor(new_shape);
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  for (unsigned i = 0; i < repeat; ++i) {
    // TODO(odashi): This calculation might generate large errors.
    unsigned offset = i % skip1 + (i / skip1) * skip2;
    float tmp = src[offset];
    for (unsigned j = 1; j < n; ++j) {
      offset += skip1;
      float arg = src[offset];
      tmp = tmp > arg
        ? tmp + std::log(1. + std::exp(arg - tmp))
        : arg + std::log(1. + std::exp(tmp - arg));
    }
    dest[i] = tmp;
  }
  return ret;
}

  Tensor CPUDevice::broadcast_impl(
    const Tensor &x, unsigned dim, unsigned size, Shape &&new_shape) {
  const unsigned repeat = x.shape().num_total_elements();
  const unsigned skip1 = new_shape.num_elements_under_rank(dim);
  const unsigned skip2 = skip1 * size;
  Tensor ret = new_tensor(new_shape);
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  for (unsigned i = 0; i < repeat; ++i) {
    unsigned offset = i % skip1 + (i / skip1) * skip2;
    float tmp = src[i];
    for (unsigned j = 0; j < size; ++j) {
      dest[offset] = tmp;
      offset += skip1;
    }
  }
  return ret;
}

Tensor CPUDevice::batch_sum_impl(const Tensor &x) {
  Tensor ret = new_tensor(x.shape().resize_batch(1));
  float *dest = DATA(ret);
  const float *src = CDATA(x);
  const unsigned bs = x.shape().batch_size();
  const unsigned size = ret.shape().num_total_elements();
  for (unsigned i = 0; i < size; ++i) {
    float temp = 0;
    for (unsigned batch = 0, pos = i; batch < bs; ++batch, pos += size) {
      temp += src[pos];
    }
    dest[i] = temp;
  }
  return ret;
}

void CPUDevice::add_gradient_impl(Tensor &a, const Tensor &b) {
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const unsigned size = sa.num_elements_per_sample();
  const unsigned bs = std::max(sa.batch_size(), sb.batch_size());
  const unsigned b_skip_d = (sa.batch_size() > 1) * size;
  const unsigned b_skip_s = (sb.batch_size() > 1) * size;
  float *dest = DATA(a);
  const float *src = CDATA(b);
  for (unsigned batch = 0; batch < bs; ++batch) {
    REPEAT_OP(i, size, dest[i] += src[i]);
    dest += b_skip_d;
    src += b_skip_s;
  }
}

void CPUDevice::add_gradient_offset_impl(
    Tensor &a, const Tensor &b, unsigned dim, unsigned offset) {
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const unsigned base = sa.num_elements_under_rank(dim);
  const unsigned span = base * sb[dim];
  const unsigned skip = base * sa[dim];
  const unsigned repeat = sa.num_elements_per_sample() / skip;
  const unsigned bs = std::max(sa.batch_size(), sb.batch_size());
  const unsigned b_skip_d =
    (sa.batch_size() > 1) * sa.num_elements_per_sample();
  const unsigned b_skip_s =
    (sb.batch_size() > 1) * sb.num_elements_per_sample();
  float *dest = DATA(a) + base * offset;
  const float *src = CDATA(b);
  for (unsigned batch = 0; batch < bs; ++batch) {
    float *dp = dest;
    const float *sp = src;
    for (unsigned i = 0; i < repeat; ++i) {
      float *ddp = dp;
      REPEAT_OP(j, span, *ddp++ += *sp++);
      dp += skip;
    }
    dest += b_skip_d;
    src += b_skip_s;
  }
}

void CPUDevice::add_gradient_sparse_impl(
    Tensor &a, const Tensor &b,
    unsigned dim, const std::vector<unsigned>& ids) {
  const unsigned bs = b.shape().batch_size();
  const unsigned skip_a =
    (a.shape().batch_size() > 1) * a.shape().num_elements_per_sample();
  const unsigned skip_i = ids.size() > 1;
  const unsigned base = b.shape().num_elements_under_rank(dim);
  const unsigned skip = base * a.shape()[dim];
  const unsigned repeat = b.shape().num_elements_per_sample() / base;
  const float *src = CDATA(b);
  for (unsigned batch = 0; batch < bs; ++batch) {
    float *dest = DATA(a) + batch * skip_a + base * ids[batch * skip_i];
    for (unsigned i = 0; i < repeat; ++i) {
      float *dp = dest;
      REPEAT_OP(j, base, *dp++ += *src++);
      dest += skip;
    }
  }
}

}  // namespace primitiv
