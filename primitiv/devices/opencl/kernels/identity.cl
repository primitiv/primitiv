#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void set_identity_kernel(
    const unsigned size, const unsigned skip, global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) py[i] = !(i % skip);
}
