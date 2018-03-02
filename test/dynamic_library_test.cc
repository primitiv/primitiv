#include <primitiv/config.h>
#include <dlls/location.h>

#include <functional>
#include <string>

#include <primitiv/dynamic_library.h>
#include <primitiv/error.h>

#include <gtest/gtest.h>

namespace primitiv {

class DynamicLibraryTest : public testing::Test {};

TEST_F(DynamicLibraryTest, CheckInitialize) {
  const std::string dirname = DLLS_DIR;
  EXPECT_NO_THROW(DynamicLibrary lib(dirname + "/basic_test.dll"));
  EXPECT_THROW(DynamicLibrary lib(dirname + "/foo"), Error);
}

TEST_F(DynamicLibraryTest, CheckGetSymbol) {
  DynamicLibrary lib(std::string(DLLS_DIR) + "/basic_test.dll");
  void *fp1 = nullptr;
  int (*fp2)(int) = nullptr;
  EXPECT_NO_THROW(fp1 = lib.get_symbol("testfunc"));
  EXPECT_NO_THROW(fp2 = lib.get_symbol<int(int)>("testfunc"));
  EXPECT_EQ(fp1, fp2);

  EXPECT_THROW(lib.get_symbol("foo"), Error);
  EXPECT_THROW(lib.get_symbol<int(int)>("foo"), Error);
}

TEST_F(DynamicLibraryTest, CheckExecute) {
  DynamicLibrary lib(std::string(DLLS_DIR) + "/basic_test.dll");
  std::function<int(int)> fp = lib.get_symbol<int(int)>("testfunc");

  for (int i : {1, 2, 3, 4}) {
    EXPECT_EQ(i * 3 + 2, fp(i));
  }
}

}  // namespace primitiv
