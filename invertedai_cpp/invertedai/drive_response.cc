#include "drive_response.h"

using json = nlohmann::json;

namespace invertedai {

DriveResponse::DriveResponse(const std::string &body_str) {

  body_json_ = nlohmann::json::parse(body_str);

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
  this->is_inside_supported_area_.clear();
  for (const auto &element : this->body_json_["is_inside_supported_area"]) {
    this->is_inside_supported_area_.push_back(element);
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
  if (this->traffic_lights_states_.has_value()) {
    this->traffic_lights_states_.value().clear();
  } else {
    this->traffic_lights_states_ = std::map<std::string, std::string>();
  }
  if (this->body_json_["traffic_lights_states"].is_null()) {
    this->traffic_lights_states_ = std::nullopt;
  } else {
    for (const auto &element : this->body_json_["traffic_lights_states"].items()) {
      this->traffic_lights_states_.value()[element.key()] = element.value();
    }
  }
  if (this->light_recurrent_states_.has_value()) {
    this->light_recurrent_states_.value().clear();
  } else {
    this->light_recurrent_states_ = std::vector<LightRecurrentState>();
  }
  if (this->body_json_["light_recurrent_states"].is_null()) {
    this->light_recurrent_states_ = std::nullopt;
  } else {
    for (const auto &element : this->body_json_["light_recurrent_states"]) {
      LightRecurrentState light_recurrent_state = {
        element[0], 
        element[1]
      };
      this->light_recurrent_states_.value().push_back(light_recurrent_state);
    }
  }
  this->birdview_.clear();
  for (const auto &element : this->body_json_["birdview"]) {
    this->birdview_.push_back(element);
  }
  this->infraction_indicators_.clear();
  for (const auto &element : this->body_json_["infraction_indicators"]) {
    InfractionIndicator infraction_indicator = {
      element[0], 
      element[1],
      element[2]
    };
    this->infraction_indicators_.push_back(infraction_indicator);
  }
  this->model_version_.clear();
  if (this->body_json_.contains("model_version") && !this->body_json_["model_version"].is_null()) {
    this->model_version_ = this->body_json_["model_version"].get<std::string>();
  } else {
      this->model_version_.clear();
  }
}
DriveResponse::DriveResponse() {
  body_json_ = nlohmann::json::object();
  agent_states_.clear();
  is_inside_supported_area_.clear();
  recurrent_states_.clear();
  traffic_lights_states_ = std::nullopt;
  light_recurrent_states_ = std::nullopt;
  birdview_.clear();
  infraction_indicators_.clear();
  model_version_.clear();
}

void DriveResponse::refresh_body_json_() {
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
  this->body_json_["is_inside_supported_area"].clear();
  for (bool is_in : this->is_inside_supported_area_) {
    this->body_json_["is_inside_supported_area"].push_back(is_in);
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
        light_recurrent_state.time_remaining
      };
      this->body_json_["light_recurrent_states"].push_back(element);
    }
  } else {
    this->body_json_["light_recurrent_states"] = nullptr;
  }
  this->body_json_["birdview"].clear();
  for (unsigned char element : this->birdview_) {
    this->body_json_["birdview"].push_back(element);
  }
  this->body_json_["infraction_indicators"].clear();
  for (InfractionIndicator infraction_indicator : this->infraction_indicators_) {
    json element = {
      infraction_indicator.collisions,
      infraction_indicator.offroad,
      infraction_indicator.wrong_way
    };
    this->body_json_["infraction_indicators"].push_back(element);
  }
  this->model_version_.clear();
  if (!this->model_version_.empty()) {
    this->body_json_["model_version"] = this->model_version_;
  } else {
      this->body_json_["model_version"] = nullptr;
  }
}

std::string DriveResponse::body_str() {
  this->refresh_body_json_();
  return this->body_json_.dump();
}

std::vector<AgentState> DriveResponse::agent_states() const {
  return this->agent_states_;
}

std::vector<bool> DriveResponse::is_inside_supported_area() const {
  return this->is_inside_supported_area_;
}

std::vector<std::vector<double>> DriveResponse::recurrent_states() const {
  return this->recurrent_states_;
}

std::optional<std::map<std::string, std::string>> DriveResponse::traffic_lights_states() const {
  return this->traffic_lights_states_;
}

std::optional<std::vector<LightRecurrentState>> DriveResponse::light_recurrent_states() const {
  return this->light_recurrent_states_;
}

std::vector<unsigned char> DriveResponse::birdview() const {
  return this->birdview_;
}

std::vector<InfractionIndicator> DriveResponse::infraction_indicators() const {
  return this->infraction_indicators_;
}

std::string DriveResponse::model_version() const {
  return this->model_version_;
}

void DriveResponse::set_agent_states(const std::vector<AgentState> &agent_states) {
  this->agent_states_ = agent_states;
}

void DriveResponse::set_is_inside_supported_area(const std::vector<bool> &is_inside_supported_area) {
  this->is_inside_supported_area_ = is_inside_supported_area;
}

void DriveResponse::set_recurrent_states(const std::vector<std::vector<double>> &recurrent_states) {
  this->recurrent_states_ = recurrent_states;
}

void DriveResponse::set_traffic_lights_states(const std::map<std::string, std::string> &traffic_lights_states) {
  this->traffic_lights_states_ = traffic_lights_states;
}

void DriveResponse::set_light_recurrent_states(const std::vector<LightRecurrentState> &light_recurrent_states) {
  this->light_recurrent_states_ = light_recurrent_states;
}

void DriveResponse::set_birdview(const std::vector<unsigned char> &birdview) {
  this->birdview_ = birdview;
}

void DriveResponse::set_infraction_indicators(const std::vector<InfractionIndicator> &infraction_indicators) {
  this->infraction_indicators_ = infraction_indicators;
}

} // namespace invertedai
