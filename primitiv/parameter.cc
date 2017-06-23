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
  em << YAML::Value << shape.batch_size();

  em << YAML::EndMap;
  return em;
}

// Operator to emit Tensor object to YAML.
YAML::Emitter &operator<<(YAML::Emitter&em, const primitiv::Tensor &tensor) {
  em << YAML::BeginMap;

  em << YAML::Key << "shape";
  em << YAML::Value << tensor.shape();
  em << YAML::Key << "value";
  const vector<float> v = tensor.to_vector();
  const unsigned s = sizeof(float) * tensor.shape().num_total_elements();
  em << YAML::Value;
  em << YAML::Binary(reinterpret_cast<const unsigned char *>(&v[0]), s);

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
    const YAML::Node &node, primitiv::Device *device) {
  primitiv::Shape shape;
  YAML::Binary data;

  for (const auto &kv : node) {
    const string key = kv.first.as<string>();
    if (key == "shape") shape = ::parse_shape(kv.second);
    else if (key == "value") data = kv.second.as<YAML::Binary>();
    else THROW_ERROR("Unknown YAML key: " << key);
  }

  const unsigned size = sizeof(float) * shape.num_total_elements();

  if (data.size() != size) {
    THROW_ERROR(
        "Data sizes mismatched. data.size(): " << std::to_string(data.size())
        << " != shape: " << shape.to_string()
        << " (size: " << std::to_string(size) << ')');
  }

  return device->new_tensor_by_array(
      shape, reinterpret_cast<const float *>(data.data()));
  
  THROW_ERROR("not implemented");
}

}  // namespace

namespace primitiv {

void Parameter::check_shape() {
  if (shape_.batch_size() > 1) {
    THROW_ERROR(
        "The batch size of the parameter shape should be 1. Given shape: "
        << shape_.to_string());
  }
}

Parameter::Parameter(
    const string &name, const Shape &shape, Device *device)
: name_(name)
, shape_(shape)
, device_(device)
, value_(device->new_tensor(shape))
, grad_(device->new_tensor(shape)) {
  check_shape();
}

Parameter::Parameter(
    const string &name, const Shape &shape, Device *device,
    const vector<float> & value)
: name_(name)
, shape_(shape)
, device_(device)
, value_(device->new_tensor(shape))
, grad_(device->new_tensor(shape)) {
  check_shape();
  reset_value(value);
}

Parameter::Parameter(
    const string &name, const Shape &shape, Device *device,
    const Initializer &init)
: name_(name)
, shape_(shape)
, device_(device)
, value_(device->new_tensor(shape))
, grad_(device->new_tensor(shape)) {
  check_shape();
  reset_value(init);
}

Parameter::Parameter(const string &name, Tensor &&value)
: name_(name)
, shape_(value.shape())
, device_(value.device())
, grad_(value.device()->new_tensor(value.shape())) {
  value_ = std::move(value);
  check_shape();
}

void Parameter::reset_value(const vector<float> &value) {
  value_.reset_by_vector(value);
}

void Parameter::reset_value(const Initializer &init) {
  init.apply(value_);
}

void Parameter::reset_gradient() {
  grad_.reset(0);
}

void Parameter::add_value(const Tensor &diff) {
  value_.add_gradient(diff);
}

void Parameter::add_gradient(const Tensor &diff) {
  grad_.add_gradient(diff);
}

void Parameter::save(const string &path) const  {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    THROW_ERROR("Could not open file: " << path);
  }

  YAML::Emitter em(ofs);
  em << YAML::BeginMap;
  em << YAML::Key << "name" << YAML::Value << name_;
  em << YAML::Key << "value" << YAML::Value << value_;
  em << YAML::EndMap;
}

Parameter Parameter::load(const string &path, Device *device) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    THROW_ERROR("Could not open file: " << path);
  }
  ifs.seekg(0, std::ios::end);
  const unsigned size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  vector<char> data(size);
  ifs.read(&data[0], size);

  YAML::Node node = YAML::Load(&data[0]);
  string name;
  Tensor value;

  for (const auto &kv : node) {
    const string key = kv.first.as<string>();
    if (key == "name") name = kv.second.as<string>();
    else if (key == "value") value = ::parse_tensor(kv.second, device);
    else THROW_ERROR("Unknown YAML key: " << key);
  }

  return Parameter(name, std::move(value));
}

}  // namespace primitiv
