#ifndef PRIMITIV_MODEL_H_
#define PRIMITIV_MODEL_H_

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace primitiv {

class Parameter;

/**
 * Set of parameters and specific algorithms.
 */
class Model {
  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;

public:
  Model() = default;
  Model(Model &&) = default;
  Model &operator=(Model &&) = default;
  virtual ~Model() = default;

  /**
   * Registers a new parameter.
   * @param name Name of the parameter.
   * @param param Reference to the parameter.
   */
  void add_parameter(const std::string &name, Parameter &param);

private:
  std::unordered_map<std::string, Parameter *> param_kv_;
  std::unordered_set<Parameter *> param_set_;  // Shortcut to find parameters.
};

}  // namespace primitiv

#endif  // PRIMITIV_MODEL_H_
