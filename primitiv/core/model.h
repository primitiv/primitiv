#ifndef PRIMITIV_CORE_MODEL_H_
#define PRIMITIV_CORE_MODEL_H_

#include <initializer_list>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <primitiv/core/error.h>
#include <primitiv/core/mixins/nonmovable.h>

namespace primitiv {

class Device;
class Parameter;

/**
 * Set of parameters and specific algorithms.
 */
class Model : mixins::Nonmovable<Model> {
public:
  Model() = default;
  virtual ~Model() = default;

  /**
   * Loads all parameters from a file.
   * @param path Path of the file.
   * @param with_stats Whether or not to load all additional statistics.
   * @param device Device object to manage parameters.
   */
  void load(const std::string &path, bool with_stats, Device *device);

  /**
   * Loads all parameters from a file.
   * @param path Path of the file.
   * @param with_stats Whether or not to load all additional statistics.
   * @param device Device object to manage parameters.
   */
  void load(const std::string &path, bool with_stats, Device &device) {
    load(path, with_stats, &device);
  }

  /**
   * Loads all parameters from a file.
   * @param path Path of the file.
   * @param with_stats Whether or not to load all additional statistics.
   */
  void load(const std::string &path, bool with_stats) {
    load(path, with_stats, nullptr);
  }

  /**
   * Loads all parameters from a file.
   * @param path Path of the file.
   */
  void load(const std::string &path) {
    load(path, true, nullptr);
  }

  /**
   * Saves all parameters to a file.
   * @param path Path of the file.
   * @param with_stats Whether or not to save all additional statistics.
   */
  void save(const std::string &path, bool with_stats) const;

  /**
   * Saves all parameters to a file.
   * @param path Path of the file.
   */
  void save(const std::string &path) const {
    save(path, true);
  }

  /**
   * Registers a new parameter.
   * @param name Name of the parameter.
   * @param param Reference to the parameter.
   * @remarks `name` should not be overlapped with all registered parameters and
   *          submodels.
   */
  void add(const std::string &name, Parameter &param);

  /**
   * Registers a new submodel.
   * @param name Name of the submodel.
   * @param model Reference to the submodel.
   * @remarks `name` should not be overlapped with all registered parameters and
   *          submodels.
   */
  void add(const std::string &name, Model &model);

  /**
   * Retrieves a parameter with specified name.
   * @param name Name of the parameter.
   * @return Const-reference of the corresponding `Parameter` object.
   * @throw primitiv::Error Parameter with `name` not found.
   */
  const Parameter &get_parameter(const std::string &name) const {
    const auto it = param_kv_.find(name);
    if (it == param_kv_.end()) {
      PRIMITIV_THROW_ERROR("Parameter with name '" << name << "' not found.");
    }
    return *it->second;
  }

  /**
   * Retrieves a parameter with specified name.
   * @param name Name of the parameter.
   * @return Reference of the corresponding `Parameter` object.
   * @throw primitiv::Error Parameter with `name` not found.
   */
  Parameter &get_parameter(const std::string &name) {
    return const_cast<Parameter &>(
        static_cast<const Model *>(this)->get_parameter(name));
  }

  /**
   * Recursively searches a parameter with specified name hierarchy.
   * @param names Name hierarchy of the parameter.
   * @return Const-reference of the corresponding `Parameter` object.
   * @throw primitiv::Error Parameter with `names` not found.
   */
  const Parameter &get_parameter(const std::vector<std::string> &names) const;

  /**
   * Recursively searches a parameter with specified name hierarchy.
   * @param names Name hierarchy of the parameter.
   * @return Const-reference of the corresponding `Parameter` object.
   * @throw primitiv::Error Parameter with `names` not found.
   */
  Parameter &get_parameter(const std::vector<std::string> &names) {
    return const_cast<Parameter &>(
        static_cast<const Model *>(this)->get_parameter(names));
  }

  /**
   * Recursively searches a parameter with specified name hierarchy.
   * @param names Name hierarchy of the parameter.
   * @return Const-reference of the corresponding `Parameter` object.
   * @throw primitiv::Error Parameter with `names` not found.
   */
  const Parameter &get_parameter(
      const std::initializer_list<std::string> names) const {
    return get_parameter(std::vector<std::string>(names));
  }

  /**
   * Recursively searches a parameter with specified name hierarchy.
   * @param names Name hierarchy of the parameter.
   * @return Const-reference of the corresponding `Parameter` object.
   * @throw primitiv::Error Parameter with `names` not found.
   */
  Parameter &get_parameter(const std::initializer_list<std::string> names) {
    return get_parameter(std::vector<std::string>(names));
  }

  /**
   * Retrieves a submodel with specified name.
   * @param name Name of the submodel.
   * @return Const-reference of the corresponding `Model` object.
   * @throw primitiv::Error Submodel with `name` not found.
   */
  const Model &get_submodel(const std::string &name) const {
    const auto it = submodel_kv_.find(name);
    if (it == submodel_kv_.end()) {
      PRIMITIV_THROW_ERROR("Submodel with name '" << name << "' not found.");
    }
    return *it->second;
  }

  /**
   * Retrieves a submodel with specified name.
   * @param name Name of the submodel.
   * @return Reference of the corresponding `Model` object.
   * @throw primitiv::Error Submodel with `name` not found.
   */
  Model &get_submodel(const std::string &name) {
    return const_cast<Model &>(
        static_cast<const Model *>(this)->get_submodel(name));
  }

  /**
   * Recursively searches a submodel with specified name hierarchy.
   * @param names Name hierarchy of the submodel.
   * @return Const-reference of the corresponding `Model` object.
   * @throw primitiv::Error Submodel with `names` not found.
   */
  const Model &get_submodel(const std::vector<std::string> &names) const;

  /**
   * Recursively searches a submodel with specified name hierarchy.
   * @param names Name hierarchy of the submodel.
   * @return Const-reference of the corresponding `Model` object.
   * @throw primitiv::Error Submodel with `names` not found.
   */
  Model &get_submodel(const std::vector<std::string> &names) {
    return const_cast<Model &>(
        static_cast<const Model *>(this)->get_submodel(names));
  }

  /**
   * Recursively searches a submodel with specified name hierarchy.
   * @param names Name hierarchy of the submodel.
   * @return Const-reference of the corresponding `Model` object.
   * @throw primitiv::Error Submodel with `names` not found.
   */
  const Model &get_submodel(
      const std::initializer_list<std::string> names) const {
    return get_submodel(std::vector<std::string>(names));
  }

  /**
   * Recursively searches a submodel with specified name hierarchy.
   * @param names Name hierarchy of the submodel.
   * @return Const-reference of the corresponding `Model` object.
   * @throw primitiv::Error Submodel with `names` not found.
   */
  Model &get_submodel(const std::initializer_list<std::string> names) {
    return get_submodel(std::vector<std::string>(names));
  }

  /**
   * Retrieves all parameters in the model.
   * @return Dictionary of parameters.
   */
  std::map<std::vector<std::string>, Parameter *> get_all_parameters() const;

  /**
   * Retrieves trainable parameters in the model.
   * @return Dictionary of parameters.
   */
  std::map<std::vector<std::string>, Parameter *> get_trainable_parameters(
      ) const {
    // NOTE(odashi):
    // Currently this function returns all parameters.
    return get_all_parameters();
  }

private:
  /**
   * Check whether specified model is contained or not in the submodel
   * hierarchy.
   * All descendant submodels will be searched by this function.
   * @param model Start point of the traversing path.
   * @return true if at least one descendant submodel is equal to `model`,
   *         false otherwise.
   */
  bool has_submodel(const Model &model) const;

  std::unordered_map<std::string, Parameter *> param_kv_;
  std::unordered_map<std::string, Model *> submodel_kv_;
  std::unordered_set<std::string> name_set_;
  std::unordered_set<Parameter *> param_set_;
  std::unordered_set<Model *> submodel_set_;

  /**
   * Searches semi-terminal submodel with specified name hierarchy.
   * @param names Name hierarchy of the submodel or parameter.
   * @return Const-reference of the corresponding semi-terminal `Model` object.
   * @throw primitiv::Error Submodel with specified hierarchy not found.
   */
  const Model &get_semiterminal(const std::vector<std::string> &names) const;
};

}  // namespace primitiv

#endif  // PRIMITIV_CORE_MODEL_H_
