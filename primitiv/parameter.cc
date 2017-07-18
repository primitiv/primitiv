#include <config.h>

#include <fstream>
#include <primitiv/device.h>
#include <primitiv/error.h>
#include <primitiv/initializer.h>
#include <primitiv/parameter.h>
#include <yaml-cpp/yaml.h>

using std::string;
using std::vector;

namespace {

// Operator to emit Shape object to YAML.
YAML::Emitter &operator<<(YAML::Emitter &em, const primitiv::Shape &shape) {
  em << YAML::BeginMap;

  em << YAML::Key << "dims";
  em << YAML::Value << YAML::BeginSeq;
  for (unsigned i = 0; i < shape.depth(); ++i) em << shape[i];
  em << YAML::EndSeq;
  em << YAML::Key << "batch";
  em << YAML::Value << shape.batch();

  em << YAML::EndMap;
  return em;
}

// Operator to emit Tensor object to YAML.
YAML::Emitter &operator<<(YAML::Emitter &em, const primitiv::Tensor &tensor) {
  em << YAML::BeginMap;

  em << YAML::Key << "valid";
  em << YAML::Value << tensor.valid();

  if (tensor.valid()) {
    em << YAML::Key << "shape";
    em << YAML::Value << tensor.shape();
    em << YAML::Key << "value";
    const vector<float> v = tensor.to_vector();
    const unsigned s = sizeof(float) * tensor.shape().size();
    em << YAML::Value;
    em << YAML::Binary(reinterpret_cast<const unsigned char *>(&v[0]), s);
  }

  em << YAML::EndMap;
  return em;
}

// Operator to emit map of Tensors to YAML.
YAML::Emitter &operator<<(
    YAML::Emitter &em,
    const std::unordered_map<std::string, primitiv::Tensor> &tensors) {
  em << YAML::BeginMap;

  for (const auto &kv : tensors) {
    em << YAML::Key << kv.first << YAML::Value << kv.second;
  }

  em << YAML::EndMap;
  return em;
}

// Function to load Shape object from YAML::Node.
primitiv::Shape parse_shape(const YAML::Node &node) {
  unsigned batch = 1;  // default
  vector<unsigned> dims;  // default

  for (const auto &kv : node) {
    const string key = kv.first.as<string>();
    if (key == "dims") dims = kv.second.as<vector<unsigned>>();
    else if (key == "batch") batch = kv.second.as<unsigned>();
    else THROW_ERROR("Unknown YAML key: " << key);
  }

  return primitiv::Shape(dims, batch);
}

// Function to load Tensor object from YAML::Node.
primitiv::Tensor parse_tensor(
    const YAML::Node &node, primitiv::Device &device) {
  // NOTE(odashi):
  // The default value `true` maintains the backward compatibility.
  bool valid = true;
  primitiv::Shape shape;
  YAML::Binary data;

  for (const auto &kv : node) {
    const string key = kv.first.as<string>();
    if (key == "valid") valid = kv.second.as<bool>();
    else if (key == "shape") shape = ::parse_shape(kv.second);
    else if (key == "value") data = kv.second.as<YAML::Binary>();
    else THROW_ERROR("Unknown YAML key: " << key);
  }

  if (!valid) return primitiv::Tensor();  // Returns invalid (empty) tensor.

  const unsigned size = sizeof(float) * shape.size();

  if (data.size() != size) {
    THROW_ERROR(
        "Data sizes mismatched. data.size(): " << std::to_string(data.size())
        << " != shape: " << shape.to_string()
        << " (size: " << std::to_string(size) << ')');
  }

  return device.new_tensor_by_array(
      shape, reinterpret_cast<const float *>(data.data()));
}

// Function to load map of Tensors from YAML::Node.
std::unordered_map<std::string, primitiv::Tensor> parse_tensor_map(
    const YAML::Node &node, primitiv::Device &device) {
  std::unordered_map<std::string, primitiv::Tensor> tensors;

  for (const auto &kv : node) {
    tensors.emplace(kv.first.as<string>(), ::parse_tensor(kv.second, device));
  }

  return tensors;
}

}  // namespace

namespace primitiv {

void Parameter::check_shape() {
  if (shape_.has_batch()) {
    THROW_ERROR(
        "The batch size of the parameter shape should be 1. Given shape: "
        << shape_.to_string());
  }
  if (value_.shape() != shape_) {
    THROW_ERROR(
        "Shape mismatched at Parameter::check_shape()."
        << " value_.shape: " << value_.shape().to_string()
        << " != expected: " << shape_.to_string());
  }
  if (grad_.shape() != shape_) {
    THROW_ERROR(
        "Shape mismatched at Parameter::check_shape()."
        << " grad_.shape: " << grad_.shape().to_string()
        << " != expected: " << shape_.to_string());
  }
}

Parameter::Parameter(
    const string &name, const Shape &shape, Device &device)
: name_(name)
, shape_(shape)
, device_(&device)
, value_(device.new_tensor(shape))
, grad_(device.new_tensor(shape)) {
  check_shape();
}

Parameter::Parameter(
    const string &name, const Shape &shape,
    const vector<float> & value,
    Device &device)
: name_(name)
, shape_(shape)
, device_(&device)
, value_(device.new_tensor(shape))
, grad_(device.new_tensor(shape)) {
  check_shape();
  reset_value(value);
}

Parameter::Parameter(
    const string &name, const Shape &shape,
    const Initializer &init,
    Device &device)
: name_(name)
, shape_(shape)
, device_(&device)
, value_(device.new_tensor(shape))
, grad_(device.new_tensor(shape)) {
  check_shape();
  reset_value(init);
}

void Parameter::initialize_by_data(
    string &&name, Tensor &&value,
    std::unordered_map<std::string, Tensor> &&stats) {
  value_ = std::move(value);  // Initializes at first.
  name_ = std::move(name);
  shape_ = value_.shape();
  device_ = &value_.device();
  grad_ = value_.device().new_tensor(value_.shape());
  stats_ = std::move(stats);
  check_shape();
}

void Parameter::reset_value(const vector<float> &value) {
  if (!valid()) THROW_ERROR("Invalid parameter.");
  value_.reset_by_vector(value);
}

void Parameter::reset_value(const Initializer &init) {
  if (!valid()) THROW_ERROR("Invalid parameter.");
  init.apply(value_);
}

void Parameter::reset_gradient() {
  if (!valid()) THROW_ERROR("Invalid parameter.");
  grad_.reset(0);
}

void Parameter::add_stats(const std::string &name, const Shape &shape) {
  if (!valid()) THROW_ERROR("Invalid parameter.");
  if (has_stats(name)) {
    THROW_ERROR("Statistics with name `" << name << "` already exists.");
  }
  stats_.emplace(std::make_pair(name, device_->new_tensor(shape)));
}

void Parameter::save(const string &path, bool with_stats) const  {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    THROW_ERROR("Could not open file: " << path);
  }

  YAML::Emitter em(ofs);
  em << YAML::BeginMap;
  em << YAML::Key << "valid" << YAML::Value << valid();
  em << YAML::Key << "name" << YAML::Value << name_;
  em << YAML::Key << "value" << YAML::Value << value_;
  if (with_stats) em << YAML::Key << "stats" << YAML::Value << stats_;
  em << YAML::EndMap;
}

Parameter Parameter::load(const string &path, bool with_stats, Device &device) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    THROW_ERROR("Could not open file: " << path);
  }
  ifs.seekg(0, std::ios::end);
  const size_t size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  std::unique_ptr<char[]> data(new char[size + 1]);
  ifs.read(data.get(), size);
  data[size] = '\0';

  YAML::Node node = YAML::Load(data.get());
  bool valid = true;  // true for backward compatibility
  string name;
  Tensor value;
  std::unordered_map<std::string, Tensor> stats;

  for (const auto &kv : node) {
    const string key = kv.first.as<string>();
    if (key == "valid") valid = kv.second.as<bool>();
    else if (key == "name") name = kv.second.as<string>();
    else if (key == "value") value = ::parse_tensor(kv.second, device);
    else if (key == "stats") {
      if (with_stats) stats = ::parse_tensor_map(kv.second, device);
    }
    else THROW_ERROR("Unknown YAML key: " << key);
  }

  Parameter param;
  if (valid) {
    param.initialize_by_data(
        std::move(name), std::move(value), std::move(stats));
  }
  return param;
}

}  // namespace primitiv
