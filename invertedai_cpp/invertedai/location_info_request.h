#ifndef LOCATION_INFO_REQUEST_H
#define LOCATION_INFO_REQUEST_H

#include "externals/json.hpp"

#include "data_utils.h"

using json = nlohmann::json;

namespace invertedai {

class LocationInfoRequest {
private:
  std::string location_;
  bool include_map_source_;
  json body_json_;

  void refresh_body_json_();

public:
  LocationInfoRequest(const std::string &body_str);
  /**
   * Serialize all the fields into a string.
   */
  std::string body_str();
  /**
   * Return a query string containing the (key, value) pairs,
   * which can be concatenated to the url.
   */
  const std::string url_query_string() const;

  /**
   * Get location string in IAI format.
   */
  std::string location() const;
  /**
   * Check whether include the map source.
   */
  bool include_map_source() const;

  /**
   * Set location string in IAI format.
   */
  void set_location(const std::string &location);
  /**
   * Set whether include the map source.
   */
  void set_include_map_source(bool include_map_source);
};

} // namespace invertedai

#endif
