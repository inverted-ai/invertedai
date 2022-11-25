#ifndef INITIALIZE_RESPONSE_H
#define INITIALIZE_RESPONSE_H

#include "data_utils.h"
#include "externals/json.hpp"

#include <vector>

using json = nlohmann::json;

namespace invertedai {

class InitializeResponse {
private:
  std::vector<AgentState> agent_states_;
  std::vector<AgentAttributes> agent_attributes_;
  std::vector<std::vector<double>> recurrent_states_;
  std::vector<unsigned char> birdview_;
  std::vector<InfractionIndicator> infraction_indicators_;
  json body_json_;

  void refresh_body_json_();

public:
  InitializeResponse(const std::string &body_str);
  std::string body_str();

  std::vector<AgentState> agent_states() const;
  std::vector<AgentAttributes> agent_attributes() const;
  std::vector<std::vector<double>> recurrent_states() const;
  std::vector<unsigned char> birdview() const;
  std::vector<InfractionIndicator> infraction_indicators() const;

  void set_agent_states(const std::vector<AgentState> &agent_states);
  void
  set_agent_attributes(const std::vector<AgentAttributes> &agent_attributes);
  void set_recurrent_states(
      const std::vector<std::vector<double>> &recurrent_states);
  void set_birdview(const std::vector<unsigned char> &birdview);
  void set_infraction_indicators(
      const std::vector<InfractionIndicator> &infraction_indicators);
};

} // namespace invertedai

#endif
