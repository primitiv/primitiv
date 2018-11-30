#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void inplace_multiply_const_kernel(
    const float k, const unsigned size, global float *px) {
  const unsigned i = get_global_id(0);
  if (i < size) px[i] *= k;
}
