#include "location_info_request.h"

using json = nlohmann::json;

namespace invertedai {

LocationInfoRequest::LocationInfoRequest(const std::string &body_str) {
  this->body_json_ = json::parse(body_str);

  this->location_ = this->body_json_["location"];
  this->include_map_source_ =
      this->body_json_["include_map_source"].is_boolean()
          ? this->body_json_["include_map_source"].get<bool>()
          : false;
}

void LocationInfoRequest::refresh_body_json_() {
  this->body_json_["location"] = this->location_;
  this->body_json_["include_map_source"] = this->include_map_source_;
}

std::string LocationInfoRequest::body_str() {
  this->refresh_body_json_();
  return this->body_json_.dump();
}

const std::string LocationInfoRequest::url_query_string() const {
  return "?location=" + this->location_ + "&include_map_source=" +
         (this->include_map_source_ ? "true" : "false");
}

std::string LocationInfoRequest::location() const { return this->location_; }

bool LocationInfoRequest::include_map_source() const {
  return this->include_map_source_;
}

void LocationInfoRequest::set_location(const std::string &location) {
  this->location_ = location;
}

void LocationInfoRequest::set_include_map_source(bool include_map_source) {
  this->include_map_source_ = include_map_source;
}

} // namespace invertedai
