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
  std::optional<int> rendering_fov_;
  std::optional<std::pair<double, double>> rendering_center_;
  json body_json_;

  void refresh_body_json_();

public:
  /**
   * A request sent to receive an LocationInfoResponse from the API.
   */
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
   * Get the location string in IAI format.
   */
  std::optional<std::string> location() const;
  /**
   * Check whether include the map source.
   */
  bool include_map_source() const;
  /**
   * Get the fov for both x and y axis for the rendered birdview in meters.
   */
  std::optional<int> rendering_fov() const;
  /**
   * Get the center coordinates for the rendered birdview.
   */
  std::optional<std::pair<double, double>> rendering_center() const;
  /**
   * Set the location string in IAI format.
   */
  void set_location(const std::string& location);
  /**
   * Set whether include the map source.
   */
  void set_include_map_source(bool include_map_source);
  /**
   * Set the fov for both x and y axis for the rendered birdview in meters.
   */
  void set_rendering_fov(std::optional<int> rendering_fov);
  /**
   * Set the center coordinates for the rendered birdview.
   */
  void set_rendering_center(const std::optional<std::pair<double, double>> &rendering_center);
};

} // namespace invertedai

#endif
