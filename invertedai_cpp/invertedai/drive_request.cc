#include "drive_request.h"

#include "externals/json.hpp"

using json = nlohmann::json;

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
  this->random_seed_ = this->body_json_["random_seed"].is_number_integer()
                           ? this->body_json_["random_seed"].get<int>()
                           : 0;
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
  this->body_json_["random_seed"] = this->random_seed_;
}

void DriveRequest::update(const InitializeResponse &init_res) {
  this->agent_states_ = init_res.agent_states_;
  this->agent_attributes_ = init_res.agent_attributes_;
  this->recurrent_states_ = init_res.recurrent_states_;
  /*  this->body_json_["agent_states"] = init_res.body_json_["agent_states"];
    this->body_json_["agent_attributes"] =
        init_res.body_json_["agent_attributes"];
    this->body_json_["recurrent_states"] =
        init_res.body_json_["recurrent_states"];
  */
}

void DriveRequest::update(const DriveResponse &drive_res) {
  this->agent_states_ = drive_res.agent_states_;
  this->recurrent_states_ = drive_res.recurrent_states_;
  /*  this->body_json_["agent_states"] = drive_res.body_json_["agent_states"];
    this->body_json_["recurrent_states"] =
        drive_res.body_json_["recurrent_states"];
  */
}

std::string DriveRequest::body_str() {
  this->refresh_body_json_();
  return this->body_json_.dump();
}
