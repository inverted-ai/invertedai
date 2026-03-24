#ifndef INITIALIZE_RESPONSE_H
#define INITIALIZE_RESPONSE_H

#include "data_utils.h"
#include "externals/json.hpp"

#include <vector>
#include <map>
#include <optional>

using json = nlohmann::json;

namespace invertedai {

class InitializeResponse {
private:
  std::vector<AgentState> agent_states_;
  std::optional<std::vector<AgentAttributes>> agent_attributes_;
  std::vector<AgentProperties> agent_properties_;
  std::vector<std::vector<double>> recurrent_states_;
  std::optional<std::map<std::string, std::string>> traffic_lights_states_;
  std::optional<std::vector<LightRecurrentState>> light_recurrent_states_;
  std::vector<unsigned char> birdview_;
  std::vector<InfractionIndicator> infraction_indicators_;
  std::string model_version_;
  json body_json_;

  void refresh_body_json_();

public:
  InitializeResponse(const std::string &body_str);
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
   * Get static attributes for all agents.
   */
  std::optional<std::vector<AgentAttributes>> agent_attributes() const;
  /**
   * Get static properties for all agents.
   */
  std::vector<AgentProperties> agent_properties() const;
  /**
   * Get the recurrent states for all agents.
   */
  std::vector<std::vector<double>> recurrent_states() const;
  /**
   * If get_birdview was set, this contains the resulting image.
   */
  std::vector<unsigned char> birdview() const;
  /**
   * Get the states of traffic lights.
   */
  std::optional<std::map<std::string, std::string>> traffic_lights_states() const;
    /**
   * Get light recurrent states for all light groups in location,
   * each light recurrent state corresponds to one light group, 
   * and contains the state and the time remaining in that state.
   */
  std::optional<std::vector<LightRecurrentState>> light_recurrent_states() const;
  /**
   * If get_infractions was set, they are returned here.
   */
  std::vector<InfractionIndicator> infraction_indicators() const;
  /**
   * Get model version.
   */
  std::string model_version() const;

  // setters
  /**
   * Set current states of all agents. The state must include x:
   * [float], y: [float] coordinate in meters orientation: [float] in radians
   * with 0 pointing along x and pi/2 pointing along y and speed: [float] in
   * m/s.
   */
  void set_agent_states(const std::vector<AgentState> &agent_states);
  /**
   * Set static attributes for all agents.
   */
  void
  set_agent_attributes(const std::vector<AgentAttributes> &agent_attributes);
  /**
   * Set static properties for all agents.
   */
  void set_agent_properties(const std::vector<AgentProperties> &agent_properties);
  /**
   * Set the recurrent states for all agents.
   */
  void set_recurrent_states(
      const std::vector<std::vector<double>> &recurrent_states);
  /**
   * Set the states of traffic lights. If the location contains traffic lights
   * within the supported area, their current state should be provided here. Any
   * traffic light for which no state is provided will be ignored by the agents.
   */
  void set_traffic_lights_states(
      const std::map<std::string, std::string> &traffic_lights_states);
   /**
   * Set recurrent states for all light groups in location.
   */
  void set_light_recurrent_states(
      const std::vector<LightRecurrentState> &light_recurrent_states);
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
