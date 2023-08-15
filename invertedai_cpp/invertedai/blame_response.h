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
  std::vector<std::vector<unsigned char>> birdviews_;
  json body_json_;

  void refresh_body_json_();

public:
  BlameResponse(const std::string &body_str);
  /**
   * Serialize all the fields into a string.
   */
  std::string body_str();

  // getters

  std::vector<int> agents_at_fault() const;

  std::optional<float> confidence_score() const;

  std::optional<std::map<int, std::vector<std::string>>> reasons() const;

  std::vector<std::vector<unsigned char>> birdviews() const;

  // setters
  void set_agents_at_fault(const std::vector<int> &agents_at_fault);
  void set_confidence_score(std::optional<float> confidence_score);
  void set_reasons(const std::optional<std::map<int, std::vector<std::string>>> &reasons);
  void set_birdviews(const std::vector<std::vector<unsigned char>> &birdviews);
};
} // namespace invertedai

#endif
