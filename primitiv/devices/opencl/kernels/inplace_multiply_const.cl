kernel void inplace_multiply_const_kernel(
    const float k, const unsigned size, global float *px) {
  const unsigned i = get_global_id(0);
  if (i < size) px[i] *= k;
}
