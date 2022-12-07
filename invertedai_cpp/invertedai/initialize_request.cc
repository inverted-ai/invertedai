#include "initialize_request.h"

using json = nlohmann::json;

namespace invertedai {
InitializeRequest::InitializeRequest(const std::string &body_str) {
  this->body_json_ = json::parse(body_str);

  this->location_ = this->body_json_["location"];
  this->num_agents_to_spawn_ =
      this->body_json_["num_agents_to_spawn"].is_number_integer()
          ? this->body_json_["num_agents_to_spawn"].get<int>()
          : 0;
  this->conditional_agent_states_.clear();
  for (const auto &element : this->body_json_["conditional_agent_states"]) {
    AgentState agent_state = {element[0], element[1], element[2], element[3]};
    this->conditional_agent_states_.push_back(agent_state);
  }

  this->conditional_agent_attributes_.clear();
  for (const auto &element : this->body_json_["conditional_agent_attributes"]) {
    AgentAttributes agent_attribute = {element[0], element[1], element[2]};
    this->conditional_agent_attributes_.push_back(agent_attribute);
  }
  
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

std::string InitializeRequest::location() const { return this->location_; }

int InitializeRequest::num_agents_to_spawn() const {
  return this->num_agents_to_spawn_;
}

std::vector<std::vector<AgentState>> InitializeRequest::states_history() const {
  return this->states_history_;
}

std::vector<AgentState> InitializeRequest::conditional_agent_states() const {
  return this->conditional_agent_states_;
}

std::vector<AgentAttributes> InitializeRequest::conditional_agent_attributes() const {
  return this->conditional_agent_attributes_;
}

std::vector<AgentAttributes> InitializeRequest::agent_attributes() const {
  return this->agent_attributes_;
}

std::vector<std::vector<TrafficLightState>>
InitializeRequest::traffic_light_state_history() const {
  return this->traffic_light_state_history_;
}

bool InitializeRequest::get_birdview() const { return this->get_birdview_; }

bool InitializeRequest::get_infractions() const {
  return this->get_infractions_;
}

int InitializeRequest::random_seed() const { return this->random_seed_; }

void InitializeRequest::set_location(const std::string &location) {
  this->location_ = location;
}

void InitializeRequest::set_num_agents_to_spawn(int num_agents_to_spawn) {
  this->num_agents_to_spawn_ = num_agents_to_spawn;
}

void InitializeRequest::set_conditional_agent_states(const std::vector<AgentState> &conditional_agent_states) {
  this->conditional_agent_states_ = conditional_agent_states;
}
  
void InitializeRequest::set_conditional_agent_attributes(const std::vector<AgentAttributes> &conditional_agent_attributes) {
  this->conditional_agent_attributes_ = conditional_agent_attributes;
}
 
void InitializeRequest::set_states_history(
    const std::vector<std::vector<AgentState>> &states_history) {
  this->states_history_ = states_history;
}

void InitializeRequest::set_agent_attributes(
    const std::vector<AgentAttributes> &agent_attributes) {
  this->agent_attributes_ = agent_attributes;
}

void InitializeRequest::set_traffic_light_state_history(
    const std::vector<std::vector<TrafficLightState>>
        &traffic_light_state_history) {
  this->traffic_light_state_history_ = traffic_light_state_history;
}

void InitializeRequest::set_get_birdview(bool get_birdview) {
  this->get_birdview_ = get_birdview;
}

void InitializeRequest::set_get_infractions(bool get_infractions) {
  this->get_infractions_ = get_infractions;
}

void InitializeRequest::set_random_seed(int random_seed) {
  this->random_seed_ = random_seed;
}

} // namespace invertedai
