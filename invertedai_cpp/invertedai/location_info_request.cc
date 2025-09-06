#include "location_info_request.h"

using json = nlohmann::json;

namespace invertedai {

LocationInfoRequest::LocationInfoRequest(const std::string &body_str) {
  this->body_json_ = json::parse(body_str);

   this->location_ =
   (this->body_json_.contains("location") && this->body_json_["location"].is_string())
     ? std::optional<std::string>{this->body_json_["location"].get<std::string>()}
     : std::nullopt;
     std::cout << "Location: " << (this->location_.has_value() ? this->location_.value() : "null") << std::endl;
 this->timestep_ =
   (this->body_json_.contains("timestep") && this->body_json_["timestep"].is_number_integer())
     ? std::optional<int>{this->body_json_["timestep"].get<int>()}
     : std::nullopt;
     std::cout << "Timestep: " << (this->timestep_.has_value() ? std::to_string(this->timestep_.value()) : "null") << std::endl;
  this->include_map_source_ = this->body_json_["include_map_source"].is_boolean()
    ? this->body_json_["include_map_source"].get<bool>()
    : false;
    std::cout << "Include map source: " << (this->include_map_source_ ? "true" : "false") << std::endl;
  this->rendering_fov_ = this->body_json_["rendering_fov"].is_number_integer()
    ? std::optional<int>{this->body_json_["rendering_fov"].get<int>()}
    : std::nullopt;
  this->rendering_center_= this->body_json_["rendering_center"].is_null()
    ? std::nullopt
    : std::optional<std::pair<double, double>>{this->body_json_["rendering_center"]};
    std::cout << "Rendering fov: " << (this->rendering_fov_.has_value() ? std::to_string(this->rendering_fov_.value()) : "null") << std::endl;
}

void LocationInfoRequest::refresh_body_json_() {
  if (this->location_.has_value()) {
    this->body_json_["location"] = this->location_.value();
  } else {
    this->body_json_.erase("location");   
  }
  this->body_json_["timestep"] = this->timestep_.has_value() ? this->timestep_.value() : 0;
  this->body_json_["include_map_source"] = this->include_map_source_;
  if (this->rendering_fov_.has_value()) {
    this->body_json_["rendering_fov"] = this->rendering_fov_.value();
  } else {
    this->body_json_["rendering_fov"] = nullptr;
  }
  if (this->rendering_center_.has_value()) {
    this->body_json_["rendering_center"] = this->rendering_center_.value();
  } else {
    this->body_json_["rendering_center"] = nullptr;
  }
}

std::string LocationInfoRequest::body_str() {
  this->refresh_body_json_();
  return this->body_json_.dump();
}

const std::string LocationInfoRequest::url_query_string() const {
  return this->location_.has_value()
  ? "?location=" + 
    this->location_.value() + 
    "&include_map_source=" +
    (this->include_map_source_ ? "true" : "false")
  : "" +
    (
      this->rendering_fov_.has_value()
        ? "&rendering_fov=" + std::to_string(this->rendering_fov_.value())
        : ""
    ) +
    (
      this->rendering_center_.has_value()
        ? "&rendering_center=" + std::to_string(this->rendering_center_.value().first) + "," + std::to_string(this->rendering_center_.value().second)
        : ""
    );
}

std::optional<std::string> LocationInfoRequest::location() const { 
  return this->location_; 
}

std::optional<int> LocationInfoRequest::timestep() const {
  return this->timestep_;
}

bool LocationInfoRequest::include_map_source() const {
  return this->include_map_source_;
}

std::optional<int> LocationInfoRequest::rendering_fov() const {
  return this->rendering_fov_;
}

std::optional<std::pair<double, double>> LocationInfoRequest::rendering_center() const {
  return this->rendering_center_;
}

void LocationInfoRequest::set_location(const std::optional<std::string> &location) {
  if (location.has_value()) {
    this->body_json_["location"] = location.value();
  } else {
      this->body_json_.erase("location");
  }
}

void LocationInfoRequest::set_location(const std::string& location) {
  this->location_ = location;
}

void LocationInfoRequest::set_timestep(const std::optional<int> &timestep) {
  this->timestep_ = timestep;
  if (timestep.has_value()) {
      this->body_json_["timestep"] = timestep.value();
  } else {
      this->body_json_["timestep"] = nullptr;
  }
}

void LocationInfoRequest::set_include_map_source(bool include_map_source) {
  this->include_map_source_ = include_map_source;
}

void LocationInfoRequest::set_rendering_fov(std::optional<int> rendering_fov) {
  this->rendering_fov_ = rendering_fov;
}

void LocationInfoRequest::set_rendering_center(const std::optional<std::pair<double, double>> &rendering_center) {
  this->rendering_center_ = rendering_center;
}

} // namespace invertedai
