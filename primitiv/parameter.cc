#include <config.h>

#include <fstream>
#include <primitiv/error.h>
#include <primitiv/initializer.h>
#include <primitiv/messages.pb.h>
#include <primitiv/operators.h>
#include <primitiv/parameter.h>

using std::string;
using std::vector;

namespace {

// Stores Shape data to the proto message.
void store_shape(const primitiv::Shape &src, primitiv::messages::Shape &dest) {
  dest.Clear();

  auto &dims = *dest.mutable_dims();
  const unsigned dims_size = src.depth();
  dims.Reserve(dims_size);
  for (unsigned i = 0; i < dims_size; ++i) {
    dims.AddAlreadyReserved(src[i]);
  }
  dest.set_batch(src.batch());
}

// Stores Tensor data to the proto message.
void store_tensor(const primitiv::Tensor &src, primitiv::messages::Tensor &dest) {
  if (!src.valid()) THROW_ERROR("Attempted to save an invalid Tensor object.");

  dest.Clear();

  ::store_shape(src.shape(), *dest.mutable_shape());
  auto &data = *dest.mutable_data();
  const vector<float> src_data = src.to_vector();
  const unsigned data_size = src_data.size();
  data.Reserve(data_size);
  for (unsigned i = 0; i < data_size; ++i) {
    data.AddAlreadyReserved(src_data[i]);
  }
}

// Parses Shape data in the proto message.
primitiv::Shape parse_shape(const primitiv::messages::Shape &src) {
  return primitiv::Shape(
      vector<unsigned>(src.dims().begin(), src.dims().end()),
      src.batch());
}

// Parses Tensor data in the proto message.
primitiv::Tensor parse_tensor(const primitiv::messages::Tensor &src, primitiv::Device &device) {
  if (!src.has_shape()) {
    THROW_ERROR("Invalid Tensor message: message has no 'shape' member.");
  }

  primitiv::Shape shape = ::parse_shape(src.shape());
  if (static_cast<unsigned>(src.data_size()) != shape.size()) {
    THROW_ERROR(
        "Invalid Tensor message: data sizes mismatched."
        << " src.data_size(): " << std::to_string(src.data_size())
        << " != shape: " << shape.to_string()
        << " (size: " << std::to_string(shape.size()) << ')');
  }

  return device.new_tensor_by_array(shape, src.data().data());
}

void check_shape(
    const primitiv::Tensor &value,
    const primitiv::Tensor &grad) {
  const primitiv::Shape &sv = value.shape();
  const primitiv::Shape &sg = grad.shape();
  if (sv != sg) {
    THROW_ERROR(
        "Shape mismatched between value and gradient. value: "
        << sv.to_string() << ", gradient: " << sg.to_string());
  }
  if (sv.has_batch()) {
    THROW_ERROR(
        "The batch size of the parameter should be 1. shape: "
        << sv.to_string());
  }
}

}  // namespace

namespace primitiv {

Parameter::Parameter(
    const Shape &shape,
    const vector<float> & value,
    Device &device)
: shape_(shape)
, device_(&device)
, value_(operators::input<Tensor>(shape, value, device))
, grad_(operators::zeros<Tensor>(shape, device)) {
  ::check_shape(value_, grad_);
}

Parameter::Parameter(
    const Shape &shape,
    const Initializer &init,
    Device &device)
: shape_(shape)
, device_(&device)
, value_(operators::zeros<Tensor>(shape, device))
, grad_(operators::zeros<Tensor>(shape, device)) {
  ::check_shape(value_, grad_);
  init.apply(value_);
}

void Parameter::load(const string &path, bool with_stats, Device &device) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  messages::Parameter src;

  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    THROW_ERROR("Could not open file: " << path);
  }
  if (!src.ParseFromIstream(&ifs)) {
    THROW_ERROR("Failed to read Parameter message: " << path);
  }
  if (!src.has_value()) {
    THROW_ERROR("Invalid Parameter message: message has no 'value' member.");
  }

  std::unordered_map<string, Tensor> stats;
  if (with_stats) {
    for (const auto &kv : src.stats()) {
      stats.emplace(std::make_pair(
            kv.first, ::parse_tensor(kv.second, device)));
    }
  }

  Tensor value = ::parse_tensor(src.value(), device);
  Tensor grad = operators::zeros<Tensor>(value.shape(), device);
  ::check_shape(value, grad);

  // Loading succeeded. Move all data to `this`.
  shape_ = value.shape();
  device_ = &device;
  value_ = std::move(value);
  grad_ = std::move(grad);
  stats_ = std::move(stats);
}

void Parameter::save(const string &path, bool with_stats) const  {
  if (!valid()) THROW_ERROR("Attempted to save an invalid Parameter object.");

  GOOGLE_PROTOBUF_VERIFY_VERSION;

  messages::Parameter dest;
  ::store_tensor(value_, *dest.mutable_value());
  if (with_stats) {
    auto &dest_stats = *dest.mutable_stats();
    for (const auto &kv : stats_) {
      ::store_tensor(kv.second, dest_stats[kv.first]);
    }
  }

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    THROW_ERROR("Could not open file: " << path);
  }
  if (!dest.SerializeToOstream(&ofs)) {
    THROW_ERROR("Failed to write Parameter message: " << path);
  }
}

void Parameter::reset_gradient() {
  if (!valid()) THROW_ERROR("Invalid parameter.");
  grad_.reset(0);
}

void Parameter::add_stats(const string &name, const Shape &shape) {
  if (!valid()) THROW_ERROR("Invalid parameter.");
  if (has_stats(name)) {
    THROW_ERROR("Statistics with name `" << name << "` already exists.");
  }
  stats_.emplace(
      std::make_pair(name, operators::zeros<Tensor>(shape, *device_)));
}

}  // namespace primitiv
