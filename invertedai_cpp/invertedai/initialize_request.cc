#include "initialize_request.h"

using json = nlohmann::json;

InitializeRequest::InitializeRequest(const std::string &body_str) {
  this->body_json_ = json::parse(body_str);

  this->location_ = this->body_json_["location"];
  this->num_agents_to_spawn_ =
      this->body_json_["num_agents_to_spawn"].is_number_integer()
          ? this->body_json_["num_agents_to_spawn"].get<int>()
          : 0;
  this->states_history_.clear();
  for (const auto &elements : this->body_json_["states_history"]) {
    std::vector<AgentState> agent_states;
    agent_states.clear();
    for (const auto &element : elements) {
      AgentState agent_state = {element[0], element[1], element[2], element[3]};
      agent_states.push_back(agent_state);
    }
    this->states_history_.push_back(agent_states);
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

void InitializeRequest::refresh_body_json_() {
  this->body_json_["location"] = this->location_;
  this->body_json_["num_agents_to_spawn"] = this->num_agents_to_spawn_;
  this->body_json_["states_history"].clear();
  for (const std::vector<AgentState> &agent_states : this->states_history_) {
    json elements;
    elements.clear();
    for (const AgentState &agent_state : agent_states) {
      json element = {agent_state.x, agent_state.y, agent_state.orientation,
                      agent_state.speed};
      elements.push_back(element);
    }
    this->body_json_["states_history"].push_back(elements);
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
  this->body_json_["get_birdview"] = this->get_birdview_;
  this->body_json_["get_infractions"] = this->get_infractions_;
  this->body_json_["random_seed"] = this->random_seed_;
};

std::string InitializeRequest::body_str() {
  this->refresh_body_json_();
  return this->body_json_.dump();
}

void InitializeRequest::set_agent_num(int agent_num) {
  this->num_agents_to_spawn_ = agent_num;
}

void InitializeRequest::set_location(const std::string& location) {
  this->location_ = location;
}
