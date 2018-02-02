#ifndef PRIMITIV_DEVICE_OPS_NAIVE_UTILS_H_
#define PRIMITIV_DEVICE_OPS_NAIVE_UTILS_H_

#define CDATA(x) static_cast<const float *>(get_handle(x))
#define MDATA(x) static_cast<float *>(get_mutable_handle(x))

#define REPEAT_OP(i, n, op) \
  for (std::uint32_t (i) = 0; (i) < (n); ++(i)) { (op); }

#endif  // PRIMITIV_DEVICE_OPS_NAIVE_UTILS_H_
