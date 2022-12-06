#ifndef LOCATION_INFO_RESPONSE_H
#define LOCATION_INFO_RESPONSE_H

#include <vector>

#include "externals/json.hpp"

#include "data_utils.h"

using json = nlohmann::json;

namespace invertedai {

class LocationInfoResponse {
private:
  std::string version_;
  int max_agent_number_;
  std::vector<Point2d> bounding_polygon_;
  std::vector<unsigned char> birdview_image_;
  std::string osm_map_;
  Point2d map_origin_;
  std::vector<StaticMapActor> static_actors_;
  json body_json_;

  void refresh_body_json_();

public:
  LocationInfoResponse(const std::string &body_str);
  /**
   * Serialize all the fields into a string.
   */
  std::string body_str();

  // getters
  /**
   * Get map version. Matches the version in the input location string, if one
   * was specified.
   */
  std::string version() const;
  /**
   * Get Maximum number of agents recommended in the location. Use more at your
   * own risk.
   */
  int max_agent_number() const;
  /**
   * Get convex polygon denoting the boundary of the supported area within the
   * location.
   */
  std::vector<Point2d> bounding_polygon() const;
  /**
   * Get the visualization of the location.
   */
  std::vector<unsigned char> birdview_image() const;
  /**
   * Underlying map annotation, returned if include_map_source was set.
   */
  std::string osm_map() const;
  /**
   * Get the origin of the map.
   */
  Point2d map_origin() const;
  /**
   * Lists traffic lights with their IDs and locations.
   */
  std::vector<StaticMapActor> static_actors() const;

  // setters
  /**
   * Set map version. Matches the version in the input location string, if one
   * was specified.
   */
  void set_version(const std::string &version);
  /**
   * Set maximum number of agents recommended in the location.
   */
  void set_max_agent_number(int max_agent_number);
  /**
   * Set convex polygon denoting the boundary of the supported area within the
   * location.
   */
  void set_bounding_polygon(const std::vector<Point2d> &bounding_polygon);
  /**
   * Setter for birdview_image.
   */
  void set_birdview_image(const std::vector<unsigned char> &birdview_image);
  /**
   * Setter for osm_map.
   */
  void set_osm_map(const std::string &osm_map);
  /**
   * Setter for map_origin.
   */
  void set_map_origin(const Point2d map_origin);
  /**
   * Setter for static_actors.
   */
  void set_static_actors(const std::vector<StaticMapActor> &static_actors);
};

} // namespace invertedai

#endif
