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
  /**
   * Serialize all the fields into a string.
   */
  std::string body_str();

  // getters
  /**
   * Get current states of all agents.
   * x: [float], y: [float] coordinate in meters;
   * orientation: [float] in radians with 0 pointing along x
   * and pi/2 pointing along y;
   * speed: [float] in m/s.
   */
  std::vector<AgentState> agent_states() const;
  /**
   * For each agent, get whether the predicted state is inside supported area.
   */
  std::vector<bool> is_inside_supported_area() const;
  /**
   * Get the recurrent states for all agents.
   */
  std::vector<std::vector<double>> recurrent_states() const;
  /**
   * If get_birdview was set, this contains the resulting image.
   */
  std::vector<unsigned char> birdview() const;
  /**
   * If get_infractions was set, they are returned here.
   */
  std::vector<InfractionIndicator> infraction_indicators() const;

  // setters
  /**
   * Set current states of all agents. The state must include x:
   * [float], y: [float] coordinate in meters orientation: [float] in radians
   * with 0 pointing along x and pi/2 pointing along y and speed: [float] in
   * m/s.
   */
  void set_agent_states(const std::vector<AgentState> &agent_states);
  /**
   * For each agent, set whether the predicted state is inside supported area.
   */
  void set_is_inside_supported_area(
      const std::vector<bool> &is_inside_supported_area);
  /**
   * Set the recurrent states for all agents.
   */
  void set_recurrent_states(
      const std::vector<std::vector<double>> &recurrent_states);
  /**
   * Set birdview.
   */
  void set_birdview(const std::vector<unsigned char> &birdview);
  /**
   * Set infraction_indicators.
   */
  void set_infraction_indicators(
      const std::vector<InfractionIndicator> &infraction_indicators);
};
} // namespace invertedai

#endif
