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

public:
  LocationInfoRequest(const std::string &body_str);
  void refresh_body_json_();
  std::string body_str();
  const std::string url_query_string() const;

  std::string location() const;
  bool include_map_source() const;

  void set_location(const std::string &location);
  void set_include_map_source(bool include_map_source);
};

} // namespace invertedai

#endif
