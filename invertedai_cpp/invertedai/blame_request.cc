#include "blame_request.h"

#include "externals/json.hpp"

using json = nlohmann::json;

namespace invertedai {

BlameRequest::BlameRequest(const std::string &body_str) {
  this->body_json_ = json::parse(body_str);

  this->location_ = this->body_json_["location"];
  this->colliding_agents_ =
      std::make_pair(this->body_json_["colliding_agents"][0],
                     this->body_json_["colliding_agents"][1]);
  this->agent_state_history_.clear();
  for (const auto &elements : this->body_json_["agent_state_history"]) {
    std::vector<AgentState> agent_states;
    agent_states.clear();
    for (const auto &element : elements) {
      AgentState agent_state = {element[0], element[1], element[2], element[3]};
      agent_states.push_back(agent_state);
    }
    this->agent_state_history_.push_back(agent_states);
  }
  this->agent_attributes_.clear();
  for (const auto &element : this->body_json_["agent_attributes"]) {
    AgentAttributes agent_attribute = {element[0], element[1], element[2]};
    this->agent_attributes_.push_back(agent_attribute);
  }
  this->traffic_light_state_history_.clear();
  for (const auto &elements : this->body_json_["traffic_light_state_history"]) {
    std::vector<TrafficLightState> traffic_light_states;
    traffic_light_states.clear();
    for (const auto &element : elements) {
      TrafficLightState traffic_light_state = {element[0], element[1]};
      traffic_light_states.push_back(traffic_light_state);
    }
    this->traffic_light_state_history_.push_back(traffic_light_states);
  }
  this->get_birdviews_ = this->body_json_["get_birdviews"].is_boolean()
                             ? this->body_json_["get_birdviews"].get<bool>()
                             : false;
  this->get_reasons_ = this->body_json_["get_reasons"].is_boolean()
                           ? this->body_json_["get_reasons"].get<bool>()
                           : false;
  this->get_confidence_score_ =
      this->body_json_["get_confidence_score"].is_boolean()
          ? this->body_json_["get_confidence_score"].get<bool>()
          : false;
}

void BlameRequest::refresh_body_json_() {
  this->body_json_["location"] = this->location_;
  this->body_json_["colliding_agents"] = this->colliding_agents_;
  this->body_json_["agent_state_history"].clear();
  for (const std::vector<AgentState> &agent_states :
       this->agent_state_history_) {
    json elements;
    elements.clear();
    for (const AgentState &agent_state : agent_states) {
      json element = {agent_state.x, agent_state.y, agent_state.orientation,
                      agent_state.speed};
      elements.push_back(element);
    }
    this->body_json_["agent_state_history"].push_back(elements);
  }
  this->body_json_["agent_attributes"].clear();
  for (const AgentAttributes &agent_attribute : this->agent_attributes_) {
    json element = {agent_attribute.length, agent_attribute.width,
                    agent_attribute.rear_axis_offset};
    this->body_json_["agent_attributes"].push_back(element);
  }
  this->body_json_["traffic_light_state_history"].clear();
  for (const std::vector<TrafficLightState> &traffic_light_states :
       this->traffic_light_state_history_) {
    json elements;
    elements.clear();
    for (const TrafficLightState &traffic_light_state : traffic_light_states) {
      json element = {traffic_light_state.id, traffic_light_state.value};
      elements.push_back(element);
    }
    this->body_json_["traffic_light_state_history"].push_back(elements);
  }
  this->body_json_["get_birdviews"] = this->get_birdviews_;
  this->body_json_["get_reasons"] = this->get_reasons_;
  this->body_json_["get_confidence_score"] = this->get_confidence_score_;
}

std::string BlameRequest::body_str() {
  this->refresh_body_json_();
  return this->body_json_.dump();
}

std::string BlameRequest::location() const { return this->location_; }

std::pair<int, int> BlameRequest::colliding_agents() const {
  return this->colliding_agents_;
}

std::vector<std::vector<AgentState>> BlameRequest::agent_state_history() const {
  return this->agent_state_history_;
}

std::vector<AgentAttributes> BlameRequest::agent_attributes() const {
  return this->agent_attributes_;
}

std::vector<std::vector<TrafficLightState>>
BlameRequest::traffic_light_state_history() const {
  return this->traffic_light_state_history_;
}

bool BlameRequest::get_birdviews() const { return this->get_birdviews_; }

bool BlameRequest::get_reasons() const { return this->get_reasons_; }

bool BlameRequest::get_confidence_score() const {
  return this->get_confidence_score_;
}

void BlameRequest::set_location(const std::string &location) {
  this->location_ = location;
}

void BlameRequest::set_colliding_agents(
    const std::pair<int, int> &colliding_agents) {
  this->colliding_agents_ = colliding_agents;
}

void BlameRequest::set_agent_state_history(
    const std::vector<std::vector<AgentState>> &agent_state_history) {
  this->agent_state_history_ = agent_state_history;
}

void BlameRequest::set_agent_attributes(
    const std::vector<AgentAttributes> &agent_attributes) {
  this->agent_attributes_ = agent_attributes;
}

void BlameRequest::set_traffic_light_state_history(
    const std::vector<std::vector<TrafficLightState>>
        &traffic_light_state_history) {
  this->traffic_light_state_history_ = traffic_light_state_history;
}

void BlameRequest::set_get_birdviews(bool get_birdviews) {
  this->get_birdviews_ = get_birdviews;
}

void BlameRequest::set_get_reasons(bool get_reasons) {
  this->get_reasons_ = get_reasons;
}

void BlameRequest::set_get_confidence_score(bool get_confidence_score) {
  this->get_confidence_score_ = get_confidence_score;
}
} // namespace invertedai
