#ifndef INITIALIZE_REQUEST_H
#define INITIALIZE_REQUEST_H

#include "data_utils.h"
#include "externals/json.hpp"

#include <vector>

using json = nlohmann::json;

class InitializeRequest {
public:
  std::string location_;
  int num_agents_to_spawn_;
  std::vector<std::vector<AgentState>> states_history_;
  std::vector<AgentAttributes> agent_attributes_;
  std::vector<std::vector<TrafficLightState>> traffic_light_state_history_;
  bool get_birdview_;
  bool get_infractions_;
  int random_seed_;
  json body_json_;
  InitializeRequest(const std::string &body_str);
  void refresh_body_json_();
  std::string body_str();
  void set_agent_num(int agent_num);
  void set_location(const std::string &location);
};

#endif
