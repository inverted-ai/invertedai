#ifndef INITIALIZE_REQUEST_H
#define INITIALIZE_REQUEST_H

#include "data_utils.h"
#include "externals/json.hpp"

#include <vector>

using json = nlohmann::json;

namespace invertedai {

class InitializeRequest {
private:
  std::string location_;
  int num_agents_to_spawn_;
  std::vector<std::vector<AgentState>> states_history_;
  std::vector<AgentAttributes> agent_attributes_;
  std::vector<std::vector<TrafficLightState>> traffic_light_state_history_;
  bool get_birdview_;
  bool get_infractions_;
  int random_seed_;
  json body_json_;

  void refresh_body_json_();

public:
  InitializeRequest(const std::string &body_str);
  std::string body_str();

  std::string location() const;
  int num_agents_to_spawn() const;
  std::vector<std::vector<AgentState>> states_history() const;
  std::vector<AgentAttributes> agent_attributes() const;
  std::vector<std::vector<TrafficLightState>> traffic_light_state_history() const;
  bool get_birdview() const;
  bool get_infractions() const;
  int random_seed() const;

  void set_location(const std::string &location);
  void set_num_agents_to_spawn(int num_agents_to_spawn);
  void set_states_history(
      const std::vector<std::vector<AgentState>> &states_history);
  void
  set_agent_attributes(const std::vector<AgentAttributes> &agent_attributes);
  void set_traffic_light_state_history(
      const std::vector<std::vector<TrafficLightState>>
          &traffic_light_state_history);
  void set_get_birdview(bool get_birdview);
  void set_get_infractions(bool get_infractions);
  void set_random_seed(int random_seed);
};

} // namespace invertedai

#endif
