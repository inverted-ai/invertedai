#include "location_info_response.h"

using json = nlohmann::json;

namespace invertedai {

LocationInfoResponse::LocationInfoResponse(const std::string &body_str) {
  this->body_json_ = json::parse(body_str);

  this->version_ = this->body_json_["version"];
  this->max_agent_number_ = this->body_json_["max_agent_number"].is_number_integer()
    ? this->body_json_["max_agent_number"].get<int>()
    : 0;
  this->bounding_polygon_.clear();
  for (const auto &element : this->body_json_["bounding_polygon"]) {
    Point2d point = {element[0], element[1]};
    this->bounding_polygon_.push_back(point);
  }
  this->birdview_image_.clear();
  for (auto &element : body_json_["birdview_image"]) {
    this->birdview_image_.push_back(element);
  }
  this->osm_map_ = this->body_json_["osm_map"];
  this->map_origin_ = {
    this->body_json_["map_origin"][0],
    this->body_json_["map_origin"][1]
  };
  this->static_actors_.clear();
  for (const auto &element : this->body_json_["static_actors"]) {
    std::optional<int> length = element["length"].is_number_float()
      ? std::optional<int>{element["length"]}
      : std::nullopt;
    std::optional<int> width = element["width"].is_number_float()
      ? std::optional<int>{element["width"]}
      : std::nullopt;
    std::optional<std::vector<int>> dependant = element["dependant"].is_array()
      ? std::optional<std::vector<int>>{element["dependant"]}
      : std::nullopt;
    StaticMapActor static_map_actor = {
      element["actor_id"],
      element["agent_type"],
      element["x"],
      element["y"],
      element["orientation"],
      length,
      width,
      dependant
    };
    this->static_actors_.push_back(static_map_actor);
  }
}

void LocationInfoResponse::refresh_body_json_() {
  this->body_json_["version"] = this->version_;
  this->body_json_["max_agent_number"] = this->max_agent_number_;
  this->body_json_["bounding_polygon"].clear();
  for (const Point2d &point : this->bounding_polygon_) {
    json element = {point.x, point.y};
    this->body_json_["bounding_polygon"].push_back(element);
  }
  this->body_json_["birdview_image"].clear();
  for (unsigned char element : this->birdview_image_) {
    this->body_json_["birdview_image"].push_back(element);
  }
  this->body_json_["osm_map"] = this->osm_map_;
  this->body_json_["map_origin"] = {this->map_origin_.x, this->map_origin_.y};
  this->body_json_["static_actors"].clear();
  for (const auto &static_map_actor : this->static_actors_) {
    json element;
    element["actor_id"] = static_map_actor.actor_id;
    element["agent_type"] = static_map_actor.agent_type;
    element["x"] = static_map_actor.x;
    element["y"] = static_map_actor.y;
    element["orientation"] = static_map_actor.orientation;
    if (static_map_actor.length.has_value()) {
      element["length"] = static_map_actor.length.value();
    } else {
      element["length"] = nullptr;
    }
    if (static_map_actor.width.has_value()) {
      element["width"] = static_map_actor.width.value();
    } else {
      element["width"] = nullptr;
    }
    if (static_map_actor.dependant.has_value()) {
      element["dependant"] = static_map_actor.dependant.value();
    } else {
      element["dependant"] = nullptr;
    }    this->body_json_["static_actors"].push_back(element);
  }
}

std::string LocationInfoResponse::body_str() {
  this->refresh_body_json_();
  return this->body_json_.dump();
}

std::string LocationInfoResponse::version() const { 
  return this->version_;
}

int LocationInfoResponse::max_agent_number() const {
  return this->max_agent_number_;
}

std::vector<Point2d> LocationInfoResponse::bounding_polygon() const {
  return this->bounding_polygon_;
}

std::vector<unsigned char> LocationInfoResponse::birdview_image() const {
  return this->birdview_image_;
}

std::string LocationInfoResponse::osm_map() const { 
  return this->osm_map_; 
}

Point2d LocationInfoResponse::map_origin() const { 
  return this->map_origin_; 
}

std::vector<StaticMapActor> LocationInfoResponse::static_actors() const {
  return static_actors_;
}

void LocationInfoResponse::set_version(const std::string &version) {
  this->version_ = version;
}

void LocationInfoResponse::set_max_agent_number(int max_agent_number) {
  this->max_agent_number_ = max_agent_number;
}

void LocationInfoResponse::set_bounding_polygon(const std::vector<Point2d> &bounding_polygon) {
  this->bounding_polygon_ = bounding_polygon;
}

void LocationInfoResponse::set_birdview_image(const std::vector<unsigned char> &birdview_image) {
  this->birdview_image_ = birdview_image;
}

void LocationInfoResponse::set_osm_map(const std::string &osm_map) {
  this->osm_map_ = osm_map;
}

void LocationInfoResponse::set_map_origin(const Point2d map_origin) {
  this->map_origin_ = map_origin;
}

void LocationInfoResponse::set_static_actors(const std::vector<StaticMapActor> &static_actors) {
  this->static_actors_ = static_actors;
}

} // namespace invertedai
