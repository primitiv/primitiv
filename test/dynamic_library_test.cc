#include <primitiv/config.h>
#include "dynamic_library_test_config.h"

#include <functional>
#include <string>

#include <primitiv/dynamic_library.h>
#include <primitiv/error.h>

#include <gtest/gtest.h>


namespace primitiv {

class DynamicLibraryTest : public testing::Test {};

TEST_F(DynamicLibraryTest, CheckInitialize) {
  EXPECT_NO_THROW(DynamicLibrary lib(TEST_DLL_PATH));
  EXPECT_THROW(DynamicLibrary lib(TEST_DLL_PATH + std::string("foo")), Error);
}

TEST_F(DynamicLibraryTest, CheckGetSymbol) {
  DynamicLibrary lib(TEST_DLL_PATH);
  void *fp1 = nullptr;
  int (*fp2)(int) = nullptr;
  EXPECT_NO_THROW(fp1 = lib.get_symbol("testfunc"));
  EXPECT_NO_THROW(fp2 = lib.get_symbol<int(int)>("testfunc"));
  EXPECT_EQ(fp1, fp2);

  EXPECT_THROW(lib.get_symbol("foo"), Error);
  EXPECT_THROW(lib.get_symbol<int(int)>("foo"), Error);
}

TEST_F(DynamicLibraryTest, CheckExecute) {
  DynamicLibrary lib(TEST_DLL_PATH);
  std::function<int(int)> fp = lib.get_symbol<int(int)>("testfunc");

  for (int i : {1, 2, 3, 4}) {
    EXPECT_EQ(i * 3 + 2, fp(i));
  }
}

}  // namespace primitiv
