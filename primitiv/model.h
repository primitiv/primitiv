#ifndef PRIMITIV_MODEL_H_
#define PRIMITIV_MODEL_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <primitiv/mixins.h>

namespace primitiv {

class Parameter;

/**
 * Set of parameters and specific algorithms.
 */
class Model : mixins::Nonmovable<Model> {
public:
  Model() = default;
  virtual ~Model() = default;

  /**
   * Registers a new parameter.
   * @param name Name of the parameter.
   * @param param Reference to the parameter.
   */
  void add_parameter(const std::string &name, Parameter &param);

  /**
   * Retrieves all parameters in the model which are trainable.
   * @return List of pointers of trainable parameters.
   */
  std::vector<Parameter *>get_trainable_parameters() const;

private:
  std::unordered_map<std::string, Parameter *> param_kv_;
  std::unordered_set<Parameter *> param_set_;  // Shortcut to find parameters.
};

}  // namespace primitiv

#endif  // PRIMITIV_MODEL_H_
