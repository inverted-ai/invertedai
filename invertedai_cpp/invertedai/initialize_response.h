#ifndef INITIALIZE_RESPONSE_H
#define INITIALIZE_RESPONSE_H

#include "data_utils.h"
#include "externals/json.hpp"

#include <vector>

using json = nlohmann::json;

class InitializeResponse {
public:
  std::vector<AgentState> agent_states_;
  std::vector<AgentAttributes> agent_attributes_;
  std::vector<std::vector<double>> recurrent_states_;
  std::vector<unsigned char> birdview_;
  std::vector<InfractionIndicator> infraction_indicators_;
  json body_json_;
  InitializeResponse(const std::string &body_str);
  void refresh_body_json_();
  std::string body_str();
};

#endif
