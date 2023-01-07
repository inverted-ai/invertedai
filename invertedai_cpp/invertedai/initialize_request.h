#ifndef INITIALIZE_REQUEST_H
#define INITIALIZE_REQUEST_H

#include "data_utils.h"
#include "externals/json.hpp"

#include <optional>
#include <vector>

using json = nlohmann::json;

namespace invertedai {

class InitializeRequest {
private:
  std::string location_;
  int num_agents_to_spawn_;
  std::vector<AgentState> conditional_agent_states_;
  std::vector<AgentAttributes> conditional_agent_attributes_;
  std::vector<std::vector<AgentState>> states_history_;
  std::vector<AgentAttributes> agent_attributes_;
  std::vector<std::vector<TrafficLightState>> traffic_light_state_history_;
  std::optional<std::pair<double, double>> location_of_interest_;
  bool get_birdview_;
  bool get_infractions_;
  std::optional<int> random_seed_;
  json body_json_;

  void refresh_body_json_();

public:
  InitializeRequest(const std::string &body_str);
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
   * Get how many agents will be spawned.
   */
  int num_agents_to_spawn() const;
  /**
   * Get history of agent states.
   */
  std::vector<AgentState> conditional_agent_states() const;
  /**
   * Get conditional agent states.
   */
  std::vector<AgentAttributes> conditional_agent_attributes() const;
  /**
   * Get conditional agent attributes.
   */

  std::vector<std::vector<AgentState>> states_history() const;
  /**
   * Get static attributes for all agents.
   */
  std::vector<AgentAttributes> agent_attributes() const;
  /**
   * Get history of traffic light states - the list is
   * over time, in chronological order.
   */
  std::vector<std::vector<TrafficLightState>>
  traffic_light_state_history() const;
  /**
   * Coordinates for spawning agents with the given location as center
   * instead of the default map center
   */
  std::optional<std::pair<double, double>> location_of_interest() const;
  /**
   * Check whether to return an image visualizing the simulation state.
   */
  bool get_birdview() const;
  /**
   * Check whether to get predicted agent states for infractions.
   */
  bool get_infractions() const;
  /**
   * Get random_seed, which controls the stochastic aspects of agent behavior
   * for reproducibility.
   */
  std::optional<int> random_seed() const;

  // setters
  /**
   * Set location string in IAI format.
   */
  void set_location(const std::string &location);
  /**
   * If states_history is not specified, this needs to be provided and
   * dictates how many agents will be spawned.
   */
  void set_num_agents_to_spawn(int num_agents_to_spawn);
  /**
   * Set history of agent states. The outer list is over agents and the
   * inner over time, in chronological order. For best results, provide at least
   * 10 historical states for each agent.
   */
  void set_conditional_agent_states(const std::vector<AgentState> &conditional_agent_states);
  /**
   * Optional conditional agent states when `agent_count` is passed. When passed,
   *  `agent_count` includes the number of conditional agents passed.
   */
  void set_conditional_agent_attributes(const std::vector<AgentAttributes> &conditional_agent_attributes);
  /**
   * Optional agent attributes when `conditional_agent_states` is passed.
   */
  void set_states_history(
      const std::vector<std::vector<AgentState>> &states_history);
  /**
   * Set static attributes for all agents.
   */
  void
  set_agent_attributes(const std::vector<AgentAttributes> &agent_attributes);
  /**
   * Set history of traffic light states - the list is
   * over time, in chronological order. Traffic light states should be provided
   * for all time steps where agent states are specified.
   */
  void set_traffic_light_state_history(
      const std::vector<std::vector<TrafficLightState>>
          &traffic_light_state_history);
  /**
   * Set coordinates for spawning agents with the given location as center
   * instead of the default map center
   */
  void set_location_of_interest(const std::optional<std::pair<double, double>>& location_of_interest);
  /**
   * Set whether to return an image visualizing the simulation state.
   * This is very slow and should only be used for debugging.
   */
  void set_get_birdview(bool get_birdview);
  /**
   * Set whether to check predicted agent states for infractions. This
   * introduces some overhead, but it should be relatively small.
   */
  void set_get_infractions(bool get_infractions);
  /**
   * Set random_seed, which controls the stochastic aspects of agent behavior
   * for reproducibility.
   */
  void set_random_seed(std::optional<int> random_seed);
};

} // namespace invertedai

#endif
