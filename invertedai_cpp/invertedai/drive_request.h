#ifndef DRIVE_REQUEST_H
#define DRIVE_REQUEST_H

#include <string>
#include <vector>

#include "externals/json.hpp"

#include "data_utils.h"
#include "drive_response.h"
#include "initialize_response.h"

using json = nlohmann::json;

class DriveRequest {
public:
  std::string location_;
  std::vector<AgentState> agent_states_;
  std::vector<AgentAttributes> agent_attributes_;
  std::vector<TrafficLightState> traffic_lights_states_;
  std::vector<std::vector<double>> recurrent_states_;
  bool get_birdview_;
  bool get_infractions_;
  int random_seed_;
  json body_json_;

  DriveRequest(const std::string &body_str);
  void update(const InitializeResponse &init_res);
  void update(const DriveResponse &drive_res);
  void refresh_body_json_();
  std::string body_str();
};

#endif
