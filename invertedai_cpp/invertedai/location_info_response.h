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
  std::string body_str();

  std::string version() const;
  int max_agent_number() const;
  std::vector<Point2d> bounding_polygon() const;
  std::vector<unsigned char> birdview_image() const;
  std::string osm_map() const;
  Point2d map_origin() const;
  std::vector<StaticMapActor> static_actors() const;

  void set_version(const std::string &version);
  void set_max_agent_number(int max_agent_number);
  void set_bounding_polygon(const std::vector<Point2d> &bounding_polygon);
  void set_birdview_image(const std::vector<unsigned char> &birdview_image);
  void set_osm_map(const std::string &osm_map);
  void set_map_origin(const Point2d map_origin);
  void set_static_actors(const std::vector<StaticMapActor> &static_actors);
};

}

#endif
