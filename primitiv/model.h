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
   * @remarks `name` should not be overlapped with all registered parameters and
   *          submodels.
   */
  void add_parameter(const std::string &name, Parameter &param);

  /**
   * Registers a new submodel.
   * @param name Name of the submodel.
   * @param model Reference to the submodel.
   * @remarks `name` should not be overlapped with all registered parameters and
   *          submodels.
   */
  void add_submodel(const std::string &name, Model &model);

  /**
   * Retrieves all parameters in the model which are trainable.
   * @return List of pointers of trainable parameters.
   */
  std::vector<Parameter *>get_trainable_parameters() const;

  /**
   * Check whether specified model is contained or not in the submodel
   * hierarchy.
   * This function is used to detect the cycle path of submodels.
   * @param model Start point of the traversing path.
   * @return true if at least one descendant submodel is equal to `model`,
   *         false otherwise.
   */
  bool has_submodel(const Model &model) const;

private:
  std::unordered_map<std::string, Parameter *> param_kv_;
  std::unordered_map<std::string, Model *> submodel_kv_;
  std::unordered_set<std::string> name_set_;
  std::unordered_set<Parameter *> param_set_;
  std::unordered_set<Model *> submodel_set_;
};

}  // namespace primitiv

#endif  // PRIMITIV_MODEL_H_
