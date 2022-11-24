#include "location_info_response.h"

using json = nlohmann::json;

LocationInfoResponse::LocationInfoResponse(const std::string &body_str) {
  this->body_json_ = json::parse(body_str);

  this->version_ = this->body_json_["version"];
  this->max_agent_number_ =
      this->body_json_["max_agent_number"].is_number_integer()
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
  this->map_origin_ = {this->body_json_["map_origin"][0],
                       this->body_json_["map_origin"][1]};
  this->static_actors_.clear();
  for (const auto &element : this->body_json_["static_actors"]) {
    StaticMapActor static_map_actor = {element[0], element[1], element[2],
                                       element[3], element[4], element[5],
                                       element[6], element[7]};
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
    json element = {static_map_actor.actor_id,    static_map_actor.agent_type,
                    static_map_actor.x,           static_map_actor.y,
                    static_map_actor.orientation, static_map_actor.length,
                    static_map_actor.width,       static_map_actor.dependant};
    this->body_json_["static_actors"].push_back(element);
  }
}

std::string LocationInfoResponse::body_str() {
  this->refresh_body_json_();
  return this->body_json_.dump();
}
