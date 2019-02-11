kernel void permute_dims_fw_kernel(
    const global float *px, const unsigned ndims, constant unsigned *x_strides,
    constant unsigned *y_strides, const unsigned size, global float *py) {
  const unsigned i = get_global_id(0);
  const unsigned bid_z = get_group_id(1);
  const unsigned ofs = bid_z * size;
  if (i < size) {
    unsigned tmp = i;
    unsigned j = 0;
    // TODO(vbkaisetsu):
    // Implove implementation
    for (unsigned d = 0; d < ndims; ++d) {
      const unsigned p = tmp / x_strides[d];
      tmp -= p * x_strides[d];
      j += p * y_strides[d];
    }
    py[ofs + j] = px[ofs + i];
  }
}

kernel void permute_dims_bw_kernel(
    const global float *py, const unsigned ndims, constant unsigned *x_strides,
    constant unsigned *y_strides, const unsigned size, global float *px) {
  const unsigned i = get_global_id(0);
  const unsigned bid_z = get_group_id(1);
  const unsigned ofs = bid_z * size;
  if (i < size) {
    unsigned tmp = i;
    unsigned j = 0;
    // TODO(vbkaisetsu):
    // Implove implementation
    for (unsigned d = 0; d < ndims; ++d) {
      const unsigned p = tmp / x_strides[d];
      tmp -= p * x_strides[d];
      j += p * y_strides[d];
    }
    px[ofs + i] += py[ofs + j];
  }
}
