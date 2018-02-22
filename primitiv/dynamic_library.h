#ifndef PRIMITIV_DYNAMIC_LIBRARY_H_
#define PRIMITIV_DYNAMIC_LIBRARY_H_

#include <string>
#include <primitiv/mixins.h>

namespace primitiv {

/**
 * Interface of dynamic (shared) library.
 */
class DynamicLibrary : mixins::Nonmovable<DynamicLibrary> {
public:
  /**
   * Loads a new dynamic library.
   * @param path Absolute path to the library file.
   * @throw primitiv::Error Loading the library failed.
   */
  DynamicLibrary(const std::string &path);

  ~DynamicLibrary();

  /**
   * Retrieves a symbol with specific name from the library.
   * @param symbol Name of the symbol.
   * @return The address of the symbol.
   * @throw primitiv::Error Retrieving the symbol failed.
   */
  void *get_symbol(const std::string &symbol);

  /**
   * Retrieves a symbol with specific name and type.
   * @param symbol Name of the symbol.
   * @return The address of the symbol.
   * @throw primitiv::Error Retrieving the symbol failed.
   */
  template <typename T>
  T *get_symbol(const std::string &symbol) {
    return reinterpret_cast<T *>(get_symbol(symbol));
  }

private:
  std::string path_;
  void *handle_;
};

}  // namespace primitiv

#endif  // PRIMITIV_DYNAMIC_LIBRARY_H_
