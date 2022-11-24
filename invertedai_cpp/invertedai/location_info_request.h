#ifndef LOCATION_INFO_REQUEST_H
#define LOCATION_INFO_REQUEST_H

#include "externals/json.hpp"

#include "data_utils.h"

using json = nlohmann::json;

class LocationInfoRequest {
public:
  std::string location_;
  bool include_map_source_;
  json body_json_;

  LocationInfoRequest(const std::string &body_str);
  void refresh_body_json_();
  std::string body_str();
  const std::string url_query_string() const;
};

#endif
