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
  std::optional<std::vector<std::vector<TrafficLightState>>>traffic_light_state_history_;
  bool get_birdviews_;
  bool get_reasons_;
  bool get_confidence_score_;
  json body_json_;

  void refresh_body_json_();

public:
  /**
   * A request sent to receive an BlameResponse from the API.
   */
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
  /**
   * Get two agents involved in the collision.
   * These integers should correspond to the indices of
   * the relevant agents in the lists within agent_state_history.
   */
  std::pair<int, int> colliding_agents() const;
  /**
   * Lists containing AgentState objects for every agent within the scene (up to
   * 100 agents) for each time step within the relevant continuous sequence
   * immediately preceding the collision. The list of AgentState objects should
   * include the first time step of the collision and no time steps afterwards.
   * The lists of AgentState objects preceding the collision should capture
   * enough of the scenario context before the collision for BLAME to analyze
   * and assign fault. For best results it is recommended to input 20-50 time
   * steps of 0.1s each preceding the collision. Each AgentState state must
   * include x: [float], y: [float] coordinates in meters, orientation: [float]
   * in radians with 0 pointing along the positive x axis and pi/2 pointing
   * along the positive y axis, and speed: [float] in m/s.
   */
  std::vector<std::vector<AgentState>> agent_state_history() const;
  /**
   * Get static attributes for all agents.
   * Each agent requires, length: [float], width: [float], and rear_axis_offset:
   * [float] all in meters.
   */
  std::vector<AgentAttributes> agent_attributes() const;
  /**
   * Get the state history of traffic lights.
   * List of TrafficLightStatesDict objects containing the state of all traffic
   * lights for every time step. The dictionary keys are the traffic-light IDs
   * and value is the state, i.e., ‘green’, ‘yellow’, ‘red’, or None.
   */
  std::optional<std::vector<std::vector<TrafficLightState>>>
  traffic_light_state_history() const;
  /**
   * Check whether to return images visualizing the simulation state.
   */
  bool get_birdviews() const;
  /**
   * Check whether to return the reasons regarding why each agent was blamed.
   */
  bool get_reasons() const;
  /**
   * Check whether to return how confident the BLAME is in its response.
   */
  bool get_confidence_score() const;

  // setters
  /**
   * Set location string in IAI format.
   */
  void set_location(const std::string &location);
  /**
   * Set two agents involved in the collision.
   * These integers should correspond to the indices of
   * the relevant agents in the lists within agent_state_history.
   */
  void set_colliding_agents(const std::pair<int, int> &colliding_agents);
  /**
   * Set the lists containing AgentState objects for every agent within the
   * scene (up to 100 agents) for each time step within the relevant continuous
   * sequence immediately preceding the collision. The list of AgentState
   * objects should include the first time step of the collision and no time
   * steps afterwards. The lists of AgentState objects preceding the collision
   * should capture enough of the scenario context before the collision for
   * BLAME to analyze and assign fault. For best results it is recommended to
   * input 20-50 time steps of 0.1s each preceding the collision. Each
   * AgentState state must include x: [float], y: [float] coordinates in meters,
   * orientation: [float] in radians with 0 pointing along the positive x axis
   * and pi/2 pointing along the positive y axis, and speed: [float] in m/s.
   */
  void set_agent_state_history(
      const std::vector<std::vector<AgentState>> &agent_state_history);
  /**
   * Set static attributes for all agents.
   * Each agent requires, length: [float], width: [float], and rear_axis_offset:
   * [float] all in meters.
   */
  void
  set_agent_attributes(const std::vector<AgentAttributes> &agent_attributes);
  /**
   * Set the list of TrafficLightStatesDict objects containing the state of all
   * traffic lights for every time step. The dictionary keys are the
   * traffic-light IDs and value is the state, i.e., ‘green’, ‘yellow’, ‘red’,
   * or None.
   */
  void set_traffic_light_state_history(const std::optional<std::vector<std::vector<TrafficLightState>>>&traffic_light_state_history);
  /**
   * Set whether to return the images visualizing the collision case.
   */
  void set_get_birdviews(bool get_birdviews);
  /**
   * Set whether to return the reasons regarding why each agent was blamed.
   */
  void set_get_reasons(bool get_reasons);
  /**
   * Set whether to return how confident the BLAME is in its response.
   */
  void set_get_confidence_score(bool get_confidence_score);
};

} // namespace invertedai

#endif
