#ifndef INVERTEDAI_API_H
#define INVERTEDAI_API_H

#include "api.h"

namespace invertedai {

LocationInfoResponse location_info(LocationInfoRequest &location_info_request,
                                   Session *session) {
  return LocationInfoResponse(session->request(
      "location_info", "", location_info_request.url_query_string()));
}

InitializeResponse initialize(InitializeRequest &initialize_request,
                              Session *session) {
  return InitializeResponse(
      session->request("initialize", initialize_request.body_str(), ""));
}

DriveResponse drive(DriveRequest &drive_request, Session *session) {
  return DriveResponse(session->request("drive", drive_request.body_str(), ""));
}

} // namespace invertedai

# endif
