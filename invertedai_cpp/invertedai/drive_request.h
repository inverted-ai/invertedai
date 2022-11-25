#ifndef DRIVE_REQUEST_H
#define DRIVE_REQUEST_H

#include <string>
#include <vector>

#include "externals/json.hpp"

#include "data_utils.h"
#include "drive_response.h"
#include "initialize_response.h"

using json = nlohmann::json;

namespace invertedai {

class DriveRequest {
private:
  std::string location_;
  std::vector<AgentState> agent_states_;
  std::vector<AgentAttributes> agent_attributes_;
  std::vector<TrafficLightState> traffic_lights_states_;
  std::vector<std::vector<double>> recurrent_states_;
  bool get_birdview_;
  bool get_infractions_;
  int random_seed_;
  json body_json_;

  void refresh_body_json_();

public:
  DriveRequest(const std::string &body_str);
  std::string body_str();
  void update(const InitializeResponse &init_res);
  void update(const DriveResponse &drive_res);

  std::string location() const;
  std::vector<AgentState> agent_states() const;
  std::vector<AgentAttributes> agent_attributes() const;
  std::vector<TrafficLightState> traffic_lights_states() const;
  std::vector<std::vector<double>> recurrent_states() const;
  bool get_birdview() const;
  bool get_infractions() const;
  int random_seed() const;

  void set_location(const std::string &location);
  void set_agent_states(const std::vector<AgentState> &agent_states);
  void
  set_agent_attributes(const std::vector<AgentAttributes> &agent_attributes);
  void set_traffic_lights_states(
      const std::vector<TrafficLightState> &traffic_lights_states);
  void set_recurrent_states(
      const std::vector<std::vector<double>> &recurrent_states);
  void set_get_birdview(bool get_birdview);
  void set_get_infractions(bool get_infractions);
  void set_random_seed(int random_seed);
};

} // namespace invertedai

#endif
