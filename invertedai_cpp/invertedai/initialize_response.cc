#include "initialize_response.h"
#include <iostream>
using json = nlohmann::json;

namespace invertedai {

InitializeResponse::InitializeResponse(const std::string &body_str) {
  this->body_json_ = json::parse(body_str);

  this->agent_states_.clear();
  for (const auto &element : body_json_["agent_states"]) {
    AgentState agent_state = {
      element[0], 
      element[1], 
      element[2], 
      element[3]
    };
    this->agent_states_.push_back(agent_state);
  }
  if (this->body_json_["agent_attributes"].is_null()) {
    this->agent_attributes_ = std::nullopt;
  } else {
    this->agent_attributes_ = std::vector<AgentAttributes>();
    for (const auto &element : this->body_json_["agent_attributes"]) {
      AgentAttributes agent_attribute(element);
      this->agent_attributes_.value().push_back(agent_attribute);
    }
  }

  this->agent_properties_ = std::vector<AgentProperties>();
  for (const auto &element : this->body_json_["agent_properties"]) {
    AgentProperties ap;
    if (element.contains("length")) {
      ap.length = element["length"];
    }
    if (element.contains("width")) {
      ap.width = element["width"];
    }
    if (element.contains("rear_axis_offset") && !element["rear_axis_offset"].is_null()) {
      ap.rear_axis_offset = element["rear_axis_offset"];
    }
    if (element.contains("agent_type")) {
      ap.agent_type = element["agent_type"];
    }
    if (element.contains("waypoint") && !element["waypoint"].is_null() ) {
      ap.waypoint = {element["waypoint"][0], element["waypoint"][1]};
    }
    if (element.contains("max_speed") && !element["max_speed"].is_null()) {
      ap.max_speed = element["max_speed"];
    }
    this->agent_properties_.push_back(ap);
  }
  this->recurrent_states_.clear();
  for (const auto &element : body_json_["recurrent_states"]) {
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
    for (const auto &element : body_json_["traffic_lights_states"].items()) {
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
    for (const auto &element : body_json_["light_recurrent_states"]) {
      LightRecurrentState light_recurrent_state = {
        element[0], 
        element[1]
      };
      this->light_recurrent_states_.value().push_back(light_recurrent_state);
    }
  }
  this->birdview_.clear();
  for (auto &element : body_json_["birdview"]) {
    this->birdview_.push_back(element);
  }
  this->infraction_indicators_.clear();
  for (auto &element : body_json_["infraction_indicators"]) {
    InfractionIndicator infraction_indicator = {
      element[0], 
      element[1],
      element[2]
    };
    this->infraction_indicators_.push_back(infraction_indicator);
  }
  this->model_version_.clear();
  this->model_version_ = body_json_["model_version"];
}

void InitializeResponse::refresh_body_json_() {
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
  if (this->body_json_["agent_attributes"].is_null()) {
    this->agent_attributes_ = std::nullopt;
  } else {
    this->agent_attributes_ = std::vector<AgentAttributes>();
    for (const auto &element : this->body_json_["agent_attributes"]) {
      AgentAttributes agent_attribute(element);
      this->agent_attributes_.value().push_back(agent_attribute);
    }
  }
  this->body_json_["agent_properties"].clear();
  for (const AgentProperties &ap : this->agent_properties_) {
    json element;
    if (ap.length.has_value()) {
      element["length"] = ap.length.value();
    }

    if (ap.width.has_value()) {
      element["width"] = ap.width.value();
    }

    if (ap.rear_axis_offset.has_value()) {
      element["rear_axis_offset"] = ap.rear_axis_offset.value();
    }

    if (ap.agent_type.has_value()) {
      element["agent_type"] = ap.agent_type.value();
    }

    if (ap.max_speed.has_value()) {
      element["max_speed"] = ap.max_speed.value();
    }
    if (ap.waypoint.has_value()) {
      element["waypoint"] = {ap.waypoint->x, ap.waypoint->y};
    }
    this->body_json_["agent_properties"].push_back(element);
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
  this->model_version_ = body_json_["model_version"];
}

std::string InitializeResponse::body_str() {
  this->refresh_body_json_();
  return this->body_json_.dump();
}

std::vector<AgentState> InitializeResponse::agent_states() const {
  return this->agent_states_;
}

std::optional<std::vector<AgentAttributes>> InitializeResponse::agent_attributes() const {
  return this->agent_attributes_;
}

std::vector<AgentProperties> InitializeResponse::agent_properties() const {
  return this->agent_properties_;
}

std::vector<std::vector<double>> InitializeResponse::recurrent_states() const {
  return this->recurrent_states_;
}

std::optional<std::map<std::string, std::string>> InitializeResponse::traffic_lights_states() const {
  return this->traffic_lights_states_;
}

std::optional<std::vector<LightRecurrentState>> InitializeResponse::light_recurrent_states() const {
  return this->light_recurrent_states_;
}

std::vector<unsigned char> InitializeResponse::birdview() const {
  return this->birdview_;
}

std::vector<InfractionIndicator> InitializeResponse::infraction_indicators() const {
  return this->infraction_indicators_;
}

std::string InitializeResponse::model_version() const {
  return this->model_version_;
}

void InitializeResponse::set_agent_states(const std::vector<AgentState> &agent_states) {
  this->agent_states_ = agent_states;
}

void InitializeResponse::set_agent_attributes(const std::vector<AgentAttributes> &agent_attributes) {
  this->agent_attributes_ = agent_attributes;
}

void InitializeResponse::set_recurrent_states(const std::vector<std::vector<double>> &recurrent_states) {
  this->recurrent_states_ = recurrent_states;
}

void InitializeResponse::set_birdview(const std::vector<unsigned char> &birdview) {
  this->birdview_ = birdview;
}

void InitializeResponse::set_infraction_indicators(const std::vector<InfractionIndicator> &infraction_indicators) {
  this->infraction_indicators_ = infraction_indicators;
}

void InitializeResponse::set_traffic_lights_states(const std::map<std::string, std::string> &traffic_lights_states) {
  this->traffic_lights_states_ = traffic_lights_states;
} 

void InitializeResponse::set_light_recurrent_states(const std::vector<LightRecurrentState> &light_recurrent_states) {
  this->light_recurrent_states_ = light_recurrent_states;
}

} // namespace invertedai
