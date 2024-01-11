#include "drive_request.h"
#include <iostream>
#include "externals/json.hpp"

using json = nlohmann::json;

namespace invertedai {

DriveRequest::DriveRequest(const std::string &body_str) {
  this->body_json_ = json::parse(body_str);

  this->location_ = this->body_json_["location"];
  this->agent_states_.clear();
  for (const auto &element : this->body_json_["agent_states"]) {
    AgentState agent_state = {
      element[0], 
      element[1], 
      element[2], 
      element[3]
    };
    this->agent_states_.push_back(agent_state);
  }
  this->agent_attributes_.clear();
  for (const auto &element : this->body_json_["agent_attributes"]) {
    AgentAttributes agent_attribute(element);
    this->agent_attributes_.push_back(agent_attribute);
  }
  this->recurrent_states_.clear();
  for (const auto &element : this->body_json_["recurrent_states"]) {
    std::vector<double> recurrent_state;
    recurrent_state.clear();
    for (const auto &inside_element : element) {
      recurrent_state.push_back(inside_element);
    }
    this->recurrent_states_.push_back(recurrent_state);
  }
  if (this->body_json_["traffic_lights_states"].is_null()) {
    this->traffic_lights_states_ = std::nullopt;
  } else {
    if (this->traffic_lights_states_.has_value()) {
      this->traffic_lights_states_.value().clear();
    } else {
      this->traffic_lights_states_ = std::map<std::string, std::string>();
    }
    for (const auto &element : this->body_json_["traffic_lights_states"].items()) {
      this->traffic_lights_states_.value()[element.key()] = element.value();
    }
  }
  if (this->body_json_["light_recurrent_states"].is_null()) {
    this->light_recurrent_states_ = std::nullopt;
  } else {
    if (this->light_recurrent_states_.has_value()) {
      this->light_recurrent_states_.value().clear();
    } else {
      this->light_recurrent_states_ = std::vector<LightRecurrentState>();
    }
    for (const auto &element : this->body_json_["light_recurrent_states"]) {
      LightRecurrentState light_recurrent_state = {
        element[0], 
        element[1]
      };
      this->light_recurrent_states_.value().push_back(light_recurrent_state);
    }
  }
  this->get_birdview_ = this->body_json_["get_birdview"].is_boolean()
    ? this->body_json_["get_birdview"].get<bool>()
    : false;
  this->get_infractions_ = this->body_json_["get_infractions"].is_boolean()
    ? this->body_json_["get_infractions"].get<bool>()
    : false;
  this->rendering_fov_ = this->body_json_["rendering_fov"].is_number_float()
    ? std::optional<int>{this->body_json_["rendering_fov"].get<double>()}
    : std::nullopt;
  this->rendering_center_ = this->body_json_["rendering_center"].is_null()
    ? std::nullopt
    : std::optional<std::pair<double, double>>{this->body_json_["rendering_center"]};
  this->random_seed_ = this->body_json_["random_seed"].is_number_integer()
    ? std::optional<int>{this->body_json_["random_seed"].get<int>()}
    : std::nullopt;
  this->model_version_ = this->body_json_["model_version"].is_null()
    ? std::nullopt
    : std::optional<std::string>{this->body_json_["model_version"]};
}

void DriveRequest::refresh_body_json_() {
  this->body_json_["location"] = this->location_;
  this->body_json_["agent_states"].clear();
  for (const AgentState &agent_state : this->agent_states_) {
    json element = {
      agent_state.x, 
      agent_state.y, 
      agent_state.orientation,
      agent_state.speed
    };
    this->body_json_["agent_states"].push_back(element);
  }
  this->body_json_["agent_attributes"].clear();
  for (const AgentAttributes &agent_attribute : this->agent_attributes_) {
    json element = agent_attribute.toJson();
    this->body_json_["agent_attributes"].push_back(element);
  }
  this->body_json_["recurrent_states"].clear();
  for (const std::vector<double> &recurrent_state : this->recurrent_states_) {
    json elements;
    elements.clear();
    for (double element : recurrent_state) {
      elements.push_back(element);
    }
    this->body_json_["recurrent_states"].push_back(elements);
  }
  this->body_json_["traffic_lights_states"].clear();
  if (this->traffic_lights_states_.has_value()) {
    for (const auto &pair : this->traffic_lights_states_.value()) {
      this->body_json_["traffic_lights_states"][pair.first] = pair.second;
    }
  } else {
    this->body_json_["traffic_lights_states"] = nullptr;
  }
  this->body_json_["light_recurrent_states"].clear();
  if (this->light_recurrent_states_.has_value()) {
    for (const LightRecurrentState &light_recurrent_state : this->light_recurrent_states_.value()) {
      json element = {
        light_recurrent_state.state, 
        light_recurrent_state.ticks_remaining
      };
      this->body_json_["light_recurrent_states"].push_back(element);
    }
  } else {
    this->body_json_["light_recurrent_states"] = nullptr;
  }
  this->body_json_["get_birdview"] = this->get_birdview_;
  this->body_json_["get_infractions"] = this->get_infractions_;
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
  if (this->random_seed_.has_value()) {
    this->body_json_["random_seed"] = this->random_seed_.value();

  } else {
    this->body_json_["random_seed"] = nullptr;
  }
  if (this->model_version_.has_value()) {
    this->body_json_["model_version"] = this->model_version_.value();
  } else {
    this->body_json_["model_version"] = nullptr;
  }
}

void DriveRequest::update(const InitializeResponse &init_res) {
  this->agent_states_ = init_res.agent_states();
  this->agent_attributes_ = init_res.agent_attributes();
  this->recurrent_states_ = init_res.recurrent_states();
  this->light_recurrent_states_ = init_res.light_recurrent_states();
}

void DriveRequest::update(const DriveResponse &drive_res) {
  this->agent_states_ = drive_res.agent_states();
  this->recurrent_states_ = drive_res.recurrent_states();
  this->light_recurrent_states_ = drive_res.light_recurrent_states();
}

std::string DriveRequest::body_str() {
  this->refresh_body_json_();
  return this->body_json_.dump();
}

std::string DriveRequest::location() const { 
  return this->location_; 
}

std::vector<AgentState> DriveRequest::agent_states() const {
  return this->agent_states_;
}

std::vector<AgentAttributes> DriveRequest::agent_attributes() const {
  return this->agent_attributes_;
};

std::optional<std::map<std::string, std::string>> DriveRequest::traffic_lights_states() const {
  return this->traffic_lights_states_;
};

std::vector<std::vector<double>> DriveRequest::recurrent_states() const {
  return this->recurrent_states_;
};

std::optional<std::vector<LightRecurrentState>> DriveRequest::light_recurrent_states() const {
  return this->light_recurrent_states_;
};

bool DriveRequest::get_birdview() const { 
  return this->get_birdview_; 
}

bool DriveRequest::get_infractions() const { 
  return this->get_infractions_; 
}

std::optional<double> DriveRequest::rendering_fov() const {
  return this->rendering_fov_;
}

std::optional<std::pair<double, double>> DriveRequest::rendering_center() const {
  return this->rendering_center_;
}

std::optional<std::string> DriveRequest::model_version() const {
  return this->model_version_;
}

std::optional<int> DriveRequest::random_seed() const {
  return this->random_seed_;
}

void DriveRequest::set_location(const std::string &location) {
  this->location_ = location;
}

void DriveRequest::set_agent_states(const std::vector<AgentState> &agent_states) {
  this->agent_states_ = agent_states;
}

void DriveRequest::set_agent_attributes(const std::vector<AgentAttributes> &agent_attributes) {
  this->agent_attributes_ = agent_attributes;
}

void DriveRequest::set_traffic_lights_states(const std::map<std::string, std::string> &traffic_lights_states) {
  this->traffic_lights_states_ = traffic_lights_states;
}

void DriveRequest::set_light_recurrent_states(const std::vector<LightRecurrentState> &light_recurrent_states) {
  this->light_recurrent_states_ = light_recurrent_states;
}

void DriveRequest::set_recurrent_states(const std::vector<std::vector<double>> &recurrent_states) {
  this->recurrent_states_ = recurrent_states;
}

void DriveRequest::set_get_birdview(bool get_birdview) {
  this->get_birdview_ = get_birdview;
}

void DriveRequest::set_get_infractions(bool get_infractions) {
  this->get_infractions_ = get_infractions;
}

void DriveRequest::set_rendering_fov(std::optional<double> rendering_fov) {
  this->rendering_fov_ = rendering_fov;
}

void DriveRequest::set_rendering_center(const std::optional<std::pair<double, double>> &rendering_center) {
  this->rendering_center_ = rendering_center;
}

void DriveRequest::set_random_seed(std::optional<int> random_seed) {
  this->random_seed_ = random_seed;
}

void DriveRequest::set_model_version(std::optional<std::string> model_version) {
  this->model_version_ = model_version;
}

} // namespace invertedai
