#ifndef DRIVE_RESPONSE_H
#define DRIVE_RESPONSE_H

#include "data_utils.h"
#include "externals/json.hpp"

#include <string>
#include <vector>

using json = nlohmann::json;

class DriveResponse {
public:
  json body_json_;
  std::vector<AgentState> agent_states_;
  std::vector<bool> is_inside_supported_area_;
  std::vector<std::vector<double>> recurrent_states_;
  std::vector<unsigned char> birdview_;
  std::vector<InfractionIndicator> infraction_indicators_;

  DriveResponse(const std::string &body_str);
  void refresh_body_json_();
  std::string body_str();
};

#endif
