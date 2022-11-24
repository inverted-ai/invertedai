#ifndef DRIVE_RESPONSE_H
#define DRIVE_RESPONSE_H

#include "data_utils.h"
#include "externals/json.hpp"

#include <string>
#include <vector>

using json = nlohmann::json;

namespace invertedai {

class DriveResponse {
private:
  std::vector<AgentState> agent_states_;
  std::vector<bool> is_inside_supported_area_;
  std::vector<std::vector<double>> recurrent_states_;
  std::vector<unsigned char> birdview_;
  std::vector<InfractionIndicator> infraction_indicators_;
  json body_json_;

  void refresh_body_json_();

public:
  DriveResponse(const std::string &body_str);
  std::string body_str();

  std::vector<AgentState> agent_states() const;
  std::vector<bool> is_inside_supported_area() const;
  std::vector<std::vector<double>> recurrent_states() const;
  std::vector<unsigned char> birdview() const;
  std::vector<InfractionIndicator> infraction_indicators() const;

  void set_agent_states(const std::vector<AgentState> &agent_states);
  void set_is_inside_supported_area(
      const std::vector<bool> &is_inside_supported_area);
  void set_recurrent_states(
      const std::vector<std::vector<double>> &recurrent_states);
  void set_birdview(const std::vector<unsigned char> &birdview);
  void set_infraction_indicators(
      const std::vector<InfractionIndicator> &infraction_indicators);
};
} // namespace invertedai

#endif
