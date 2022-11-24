/**
 * Interface wrappers for the REST API.
 */

#include "drive_request.h"
#include "drive_response.h"
#include "initialize_request.h"
#include "initialize_response.h"
#include "location_info_request.h"
#include "location_info_response.h"
#include "session.h"

namespace invertedai {

LocationInfoResponse location_info(LocationInfoRequest &location_info_request,
                                   Session *session);

InitializeResponse initialize(InitializeRequest &initialize_request,
                              Session *session);

DriveResponse drive(DriveRequest &drive_request, Session *session);

} // namespace invertedai
