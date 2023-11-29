#ifndef BLAME_RESPONSE_H
#define BLAME_RESPONSE_H

#include "data_utils.h"
#include "externals/json.hpp"

#include <string>
#include <vector>

using json = nlohmann::json;

namespace invertedai {

class BlameResponse {
private:
  std::vector<int> agents_at_fault_;
  std::optional<float> confidence_score_;
  std::optional<std::map<int, std::vector<std::string>>> reasons_;
  std::optional<std::vector<std::vector<unsigned char>>> birdviews_;
  json body_json_;

  void refresh_body_json_();

public:
  BlameResponse(const std::string &body_str);
  /**
   * Serialize all the fields into a string.
   */
  std::string body_str();

  // getters
  /**
   * Get a vector containing all agents predicted to be at fault. If empty, BLAME
   * has predicted no agents are at fault.
   */
  std::vector<int> agents_at_fault() const;
  /**
   * Get float value between [0,1] indicating BLAME’s confidence in the response
   * where 0.0 represents the minimum confidence and 1.0 represents maximum.
   */
  std::optional<float> confidence_score() const;
  /**
   * Get a map with agent IDs as keys corresponding to “agents_at_fault”
   * paired with a list of reasons why the keyed agent is at fault (e.g.
   * traffic_light_violation).
   */
  std::optional<std::map<int, std::vector<std::string>>> reasons() const;
  /**
   * Get the images visualizing the collision case.
   */
  std::optional<std::vector<std::vector<unsigned char>>> birdviews() const;

  // setters
  /**
   * Set a vector containing all agents at fault.
   */
  void set_agents_at_fault(const std::vector<int> &agents_at_fault);
  /**
   * Set float value between [0,1] indicating BLAME’s confidence in the response
   */
  void set_confidence_score(std::optional<float> confidence_score);
  /**
   * Set a map with agent IDs as keys corresponding to “agents_at_fault”
   * paired with a list of reasons why the keyed agent is at fault (e.g.
   * traffic_light_violation).
   */
  void set_reasons(const std::optional<std::map<int, std::vector<std::string>>> &reasons);
  /**
   * Set the images visualizing the collision case.
   */
  void set_birdviews(const std::optional<std::vector<std::vector<unsigned char>>> &birdviews);
};

} // namespace invertedai

#endif
