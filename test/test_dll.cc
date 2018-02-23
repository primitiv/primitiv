// This function will be called from dynamic_library_test.cc by dynamic linking.
extern "C" int testfunc(int x) {
  return x * 3 + 2;
}
