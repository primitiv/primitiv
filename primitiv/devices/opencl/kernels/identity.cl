kernel void set_identity_kernel(
    const unsigned size, const unsigned skip, global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) py[i] = !(i % skip);
}
