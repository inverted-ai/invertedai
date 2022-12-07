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

/**
 * Wrap the REST API "location_info".
 * Provides static information about a given location.
 *
 * @param location_info_request the location_info request to send to the API
 * @param session the shared session connected with the host
 * @return the location_info response receive from the API
 * @see invertedai::initialize
 * @see invertedai::drive
 */
LocationInfoResponse location_info(LocationInfoRequest &location_info_request,
                                   Session *session);

/**
 * Wrap the REST API "initialize".
 * Initializes a simulation in a given location. Either agent_count or both
 * agent_attributes and states_history need to be provided. In the latter case,
 * the simulation is initialized with the specific history, and if traffic
 * lights are present then traffic_light_state_history should also be provided.
 * If only agent_count is specified, a new initial state is generated with the
 * requested total number of agents. Every simulation needs to start with a call
 * to this function in order to obtain correct recurrent states for drive().
 *
 * @param initialize_request the initialize request to send to the API
 * @param session the shared session connected with the host
 * @return the initialize response receive from the API
 * @see invertedai::location_info
 * @see invertedai::drive
 */
InitializeResponse initialize(InitializeRequest &initialize_request,
                              Session *session);

/**
 * Wrap the REST API "drive".
 * Drive the agents based on given situations.
 *
 * @param drive_request the initialize request to send to the API
 * @param session the shared session connected with the host
 * @return the initialize response receive from the API
 * @see invertedai::location_info
 * @see invertedai::initialize
*/
DriveResponse drive(DriveRequest &drive_request, Session *session);

} // namespace invertedai
