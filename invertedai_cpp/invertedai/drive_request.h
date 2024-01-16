#ifndef DRIVE_REQUEST_H
#define DRIVE_REQUEST_H

#include <optional>
#include <string>
#include <vector>
#include <map>

#include "externals/json.hpp"

#include "data_utils.h"
#include "drive_response.h"
#include "initialize_response.h"

using json = nlohmann::json;

namespace invertedai {

class DriveRequest {
private:
  std::string location_;
  std::vector<AgentState> agent_states_;
  std::vector<AgentAttributes> agent_attributes_;
  std::optional<std::map<std::string, std::string>> traffic_lights_states_;
  std::optional<std::vector<LightRecurrentState>> light_recurrent_states_;
  std::vector<std::vector<double>> recurrent_states_;
  bool get_birdview_;
  bool get_infractions_;
  std::optional<int> random_seed_;
  std::optional<double> rendering_fov_;
  std::optional<std::pair<double, double>> rendering_center_;
  std::optional<std::string> model_version_;
  json body_json_;

  void refresh_body_json_();

public:
  /**
   * A request sent to receive an DriveResponse from the API.
   */
  DriveRequest(const std::string &body_str);
  /**
   * Serialize all the fields into a string.
   */
  std::string body_str();

  /**
   * Update the drive request with the information(agent_states,
   * agent_attributes, recurrent_states) in the initialize response.
   */
  void update(const InitializeResponse &init_res);
  /**
   * Update the drive request with the information(agent_states,
   * recurrent_states) in the drive response.
   */
  void update(const DriveResponse &drive_res);

  // getters
  /**
   * Get location string in IAI format.
   */
  std::string location() const;
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
  std::vector<AgentAttributes> agent_attributes() const;
  /**
   * Get the states of traffic lights.
   */
  std::optional<std::map<std::string, std::string>> traffic_lights_states() const;
  /**
   * Get the recurrent states for all agents.
   */
  std::vector<std::vector<double>> recurrent_states() const;
  /**
   * Get the recurrent states for all light groups in location.
   */
  std::optional<std::vector<LightRecurrentState>> light_recurrent_states() const;
  /**
   * Check whether to return an image visualizing the simulation state.
   */
  bool get_birdview() const;
  /**
   * Check whether to check predicted agent states for infractions.
   */
  bool get_infractions() const;
  /**
   * Get the fov for both x and y axis for the rendered birdview in meters.
   */
  std::optional<double> rendering_fov() const;
  /**
   * Get the center coordinates for the rendered birdview.
   */
  std::optional<std::pair<double, double>> rendering_center() const;
  /**
   * Get random_seed.
   */
  std::optional<int> random_seed() const;
  /**
   * Get model version.
   */
  std::optional<std::string> model_version() const;

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
  void set_agent_states(const std::vector<AgentState> &agent_states);
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
  void set_traffic_lights_states(
      const std::map<std::string, std::string> &traffic_lights_states);
  /**
   * Set the recurrent states for all agents, obtained from the
   * previous call to drive() or initialize().
   */
  void set_recurrent_states(
      const std::vector<std::vector<double>> &recurrent_states);
  /**
   * Set light recurrent states for all light groups in location.
   */
  void set_light_recurrent_states(
      const std::vector<LightRecurrentState> &light_recurrent_states);
  /**
   * Set whether to return an image visualizing the simulation state.
   * This is very slow and should only be used for debugging.
   */
  void set_get_birdview(bool get_birdview);
  /**
   * Set whether to check predicted agent states for infractions.
   * This introduces some overhead, but it should be relatively small.
   */
  void set_get_infractions(bool get_infractions);
  /**
   * Set the fov for both x and y axis for the rendered birdview in meters.
   */
  void set_rendering_fov(std::optional<double> rendering_fov);
  /**
   * Set the center coordinates for the rendered birdview.
   */
  void set_rendering_center(
      const std::optional<std::pair<double, double>> &rendering_center);
  /**
   * Set random_seed, which controls the stochastic aspects of agent behavior
   * for reproducibility.
   */
  void set_random_seed(std::optional<int> random_seed);
  /**
   * Set model version. If None is passed which is by default, the best model will be used.
   */
  void set_model_version(std::optional<std::string> model_version);
};

} // namespace invertedai

#endif
