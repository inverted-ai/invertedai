/**
 * Interface wrappers for the REST API.
 */

#include "blame_request.h"
#include "blame_response.h"
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
 * @see invertedai::blame
 */
LocationInfoResponse location_info(LocationInfoRequest &location_info_request,
                                   Session *session);

/**
 * Wrap the REST API "initialize".
 * Initializes a simulation in a given location, using a combination of **user-defined** and **sampled** agents.
 * **User-defined** agents are placed in a scene first, after which a number of agents are sampled conditionally 
 * inferred from the `num_agents_to_spawn` argument.
 * If **user-defined** agents are desired, `states_history` must contain a vector of `AgentState's` of all **user-defined** 
 * agents per historical time step.
 * Any **user-defined** agent must have a corresponding fully specified static `AgentAttribute` in `agent_attributes`. 
 * Any **sampled** agents not specified in `agent_attributes` will be generated with default static attribute values however
 * **sampled** agents may be defined by specifying all static attributes or by specifying `agent_type` only. 
 * Agents are identified by their vector index, so ensure the indices of each agent match in `states_history` and
 * `agent_attributes` when applicable. 
 * If traffic lights are present in the scene, for best results their state should be specified for the current time in a 
 * `TrafficLightStatesDict`, and all historical time steps for which `states_history` is provided. It is legal to omit
 * the traffic light state specification, but the scene will be initialized as if the traffic lights were disabled.
 * Every simulation must start with a call to this function in order to obtain correct recurrent states for invertedai::drive.
 *
 * @param initialize_request the initialize request to send to the API
 * @param session the shared session connected with the host
 * @return the initialize response receive from the API
 * @see invertedai::location_info
 * @see invertedai::drive
 * @see invertedai::blame
 */
InitializeResponse initialize(InitializeRequest &initialize_request,
                              Session *session);

/**
 * Wrap the REST API "drive".
 * Drive the agents based on given situations.
 *
 * @param drive_request the drive request to send to the API
 * @param session the shared session connected with the host
 * @return the drive response receive from the API
 * @see invertedai::location_info
 * @see invertedai::initialize
 * @see invertedai::blame
*/
DriveResponse drive(DriveRequest &drive_request, Session *session);


/**
 * Wrap the REST API "blame".
 * Blame the agents at fault in the collision.
 *
 * @param blame_request the blame request to send to the API
 * @param session the shared session connected with the host
 * @return the blame response receive from the API
 * @see invertedai::location_info
 * @see invertedai::initialize
 * @see invertedai::drive
*/
BlameResponse blame(BlameRequest &blame_request, Session *session);

} // namespace invertedai
