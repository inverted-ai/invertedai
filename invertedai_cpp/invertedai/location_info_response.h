#ifndef LOCATION_INFO_RESPONSE_H
#define LOCATION_INFO_RESPONSE_H

#include <vector>

#include "externals/json.hpp"

#include "data_utils.h"

using json = nlohmann::json;

class LocationInfoResponse {
public:
  std::string version_;
  int max_agent_number_;
  std::vector<Point2d> bounding_polygon_;
  std::vector<unsigned char> birdview_image_;
  std::string osm_map_;
  Point2d map_origin_;
  std::vector<StaticMapActor> static_actors_;
  json body_json_;

  LocationInfoResponse(const std::string &body_str);
  void refresh_body_json_();
  std::string body_str();
};

#endif
