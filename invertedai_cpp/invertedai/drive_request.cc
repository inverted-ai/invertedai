#include "drive_request.h"

#include "externals/json.hpp"

using json = nlohmann::json;

namespace invertedai {

DriveRequest::DriveRequest(const std::string &body_str) {
  this->body_json_ = json::parse(body_str);

  this->location_ = this->body_json_["location"];
  this->agent_states_.clear();
  for (const auto &element : this->body_json_["agent_states"]) {
    AgentState agent_state = {element[0], element[1], element[2], element[3]};
    this->agent_states_.push_back(agent_state);
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
  this->traffic_lights_states_.clear();
  for (const auto &element : this->body_json_["traffic_lights_states"]) {
    TrafficLightState traffic_light_state = {element[0], element[1]};
    this->traffic_lights_states_.push_back(traffic_light_state);
  }
  this->get_birdview_ = this->body_json_["get_birdview"].is_boolean()
                            ? this->body_json_["get_birdview"].get<bool>()
                            : false;
  this->get_infractions_ = this->body_json_["get_infractions"].is_boolean()
                               ? this->body_json_["get_infractions"].get<bool>()
                               : false;
  this->random_seed_ =
      this->body_json_["random_seed"].is_number_integer()
          ? std::optional<int>{this->body_json_["random_seed"].get<int>()}
          : std::nullopt;
}

void DriveRequest::refresh_body_json_() {
  this->body_json_["location"] = this->location_;
  this->body_json_["agent_states"].clear();
  for (const AgentState &agent_state : this->agent_states_) {
    json element = {agent_state.x, agent_state.y, agent_state.orientation,
                    agent_state.speed};
    this->body_json_["agent_states"].push_back(element);
  }
  this->body_json_["agent_attributes"].clear();
  for (const AgentAttributes &agent_attribute : this->agent_attributes_) {
    json element = {agent_attribute.length, agent_attribute.width,
                    agent_attribute.rear_axis_offset};
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
  for (const TrafficLightState &traffic_light_state :
       this->traffic_lights_states_) {
    json element = {traffic_light_state.id, traffic_light_state.value};
    this->body_json_["traffic_lights_states"].push_back(element);
  }
  this->body_json_["get_birdview"] = this->get_birdview_;
  this->body_json_["get_infractions"] = this->get_infractions_;
  if (this->random_seed_.has_value()) {
    this->body_json_["random_seed"] = this->random_seed_.value();

  } else {
    this->body_json_["random_seed"] = nullptr;
  }
}

void DriveRequest::update(const InitializeResponse &init_res) {
  this->agent_states_ = init_res.agent_states();
  this->agent_attributes_ = init_res.agent_attributes();
  this->recurrent_states_ = init_res.recurrent_states();
}

void DriveRequest::update(const DriveResponse &drive_res) {
  this->agent_states_ = drive_res.agent_states();
  this->recurrent_states_ = drive_res.recurrent_states();
}

std::string DriveRequest::body_str() {
  this->refresh_body_json_();
  return this->body_json_.dump();
}

std::string DriveRequest::location() const { return this->location_; }

std::vector<AgentState> DriveRequest::agent_states() const {
  return this->agent_states_;
}

std::vector<AgentAttributes> DriveRequest::agent_attributes() const {
  return this->agent_attributes_;
};

std::vector<TrafficLightState> DriveRequest::traffic_lights_states() const {
  return this->traffic_lights_states_;
};

std::vector<std::vector<double>> DriveRequest::recurrent_states() const {
  return this->recurrent_states_;
};

bool DriveRequest::get_birdview() const { return this->get_birdview_; }

bool DriveRequest::get_infractions() const { return this->get_infractions_; }

std::optional<int> DriveRequest::random_seed() const {
  return this->random_seed_;
}

void DriveRequest::set_location(const std::string &location) {
  this->location_ = location;
}

void DriveRequest::set_agent_states(
    const std::vector<AgentState> &agent_states) {
  this->agent_states_ = agent_states;
}

void DriveRequest::set_agent_attributes(
    const std::vector<AgentAttributes> &agent_attributes) {
  this->agent_attributes_ = agent_attributes;
}

void DriveRequest::set_traffic_lights_states(
    const std::vector<TrafficLightState> &traffic_lights_states) {
  this->traffic_lights_states_ = traffic_lights_states;
}

void DriveRequest::set_recurrent_states(
    const std::vector<std::vector<double>> &recurrent_states) {
  this->recurrent_states_ = recurrent_states;
}

void DriveRequest::set_get_birdview(bool get_birdview) {
  this->get_birdview_ = get_birdview;
}

void DriveRequest::set_get_infractions(bool get_infractions) {
  this->get_infractions_ = get_infractions;
}

void DriveRequest::set_random_seed(std::optional<int> random_seed) {
  this->random_seed_ = random_seed;
}

} // namespace invertedai
