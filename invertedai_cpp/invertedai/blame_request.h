#ifndef BLAME_REQUEST_H
#define BLAME_REQUEST_H

#include <optional>
#include <string>
#include <vector>

#include "externals/json.hpp"

#include "data_utils.h"

using json = nlohmann::json;

namespace invertedai {

class BlameRequest {
private:
  std::string location_;
  std::pair<int, int> colliding_agents_;
  std::vector<std::vector<AgentState>> agent_state_history_;
  std::vector<AgentAttributes> agent_attributes_;
  std::vector<std::vector<TrafficLightState>> traffic_light_state_history_;
  bool get_birdviews_;
  bool get_reasons_;
  bool get_confidence_score_;
  json body_json_;

  void refresh_body_json_();

public:
  BlameRequest(const std::string &body_str);
  /**
   * Serialize all the fields into a string.
   */
  std::string body_str();

  // getters
  /**
   * Get location string in IAI format.
   */
  std::string location() const;

  std::pair<int, int> colliding_agents() const;
  /**
   * Get current states of all agents.
   * x: [float], y: [float] coordinate in meters;
   * orientation: [float] in radians with 0 pointing along x
   * and pi/2 pointing along y;
   * speed: [float] in m/s.
   */
  std::vector<std::vector<AgentState>> agent_state_history() const;
  /**
   * Get static attributes for all agents.
   */
  std::vector<AgentAttributes> agent_attributes() const;
  /**
   * Get the states of traffic lights.
   */
  std::vector<std::vector<TrafficLightState>>
  traffic_light_state_history() const;
  /**
   * Check whether to return an image visualizing the simulation state.
   */
  bool get_birdviews() const;
  /**
   * Check whether to check predicted agent states for infractions.
   */
  bool get_reasons() const;
  /**
   * Check whether to check predicted agent states for infractions.
   */
  bool get_confidence_score() const;

  // setters
  /**
   * Set location string in IAI format.
   */
  void set_location(const std::string &location);
  /**
   * Set current states of all agents. The state must include x:
   * [float], y: [float] coordinate in meters orientation: [float] in radians
   * with 0 pointing along x and pi/2 pointing along y and speed: [float] in
   * m/s.
   */
  void set_colliding_agents(const std::pair<int, int> &colliding_agents);

  void set_agent_state_history(
      const std::vector<std::vector<AgentState>> &agent_state_history);
  /**
   * Set static attributes for all agents.
   */
  void
  set_agent_attributes(const std::vector<AgentAttributes> &agent_attributes);
  /**
   * Set the states of traffic lights. If the location contains traffic lights
   * within the supported area, their current state should be provided here. Any
   * traffic light for which no state is provided will be ignored by the agents.
   */
  void set_traffic_light_state_history(
      const std::vector<std::vector<TrafficLightState>> &traffic_light_state_history);
  /**
   * Set whether to return an image visualizing the simulation state.
   * This is very slow and should only be used for debugging.
   */
  void set_get_birdviews(bool get_birdviews);
  /**
   * Check whether to check predicted agent states for infractions.
   */
  void set_get_reasons(bool get_reasons);
  /**
   * Check whether to check predicted agent states for infractions.
   */
  void set_get_confidence_score(bool get_confidence_score);
};

} // namespace invertedai

#endif
