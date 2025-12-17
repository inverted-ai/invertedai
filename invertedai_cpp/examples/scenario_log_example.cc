#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <optional>
#include <opencv2/opencv.hpp>
#include <utility>
#include <ostream>


#include "invertedai/api.h"
#include "invertedai/session.h"
#include "invertedai/location_info_request.h"
#include "invertedai/location_info_response.h"
#include "invertedai/initialize_request.h"
#include "invertedai/initialize_response.h"
#include "invertedai/drive_request.h"
#include "invertedai/drive_response.h"
#include "invertedai/visualize.h"

using namespace invertedai;

/*                                                                                 
            HOW TO RUN EXECUTABLE:

            1. cd into invertedai_cpp folder

            2. Join docker:
                docker compose build
                docker compose run --rm dev 

            3. Export your API key in the docker:
                export IAI_API_KEY="your_key_here"
            
            4. Build:
                bazel build //examples:scenario_log_example

            5. To run:
                ./bazel-bin/examples/scenario_log_example --rollout_length 100

*/
const int TIMESTEP_TO_BRANCH_FROM = 10;
int main(int argc, char** argv) {
    int NEW_ROLLOUT_LENGTH = 50; // length of new rollout after branching from json
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--rollout_length") {
            NEW_ROLLOUT_LENGTH = std::stoi(argv[++i]);
        }
    }
    const std::string API_KEY = getenv("IAI_API_KEY"); 
    LogReader log_reader("examples/carla_Town10HD_log.json");
    boost::asio::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    invertedai::Session session(ioc, ctx);
    session.set_api_key(API_KEY);
    session.connect();

    const std::string location = log_reader.get_location();
    int fov;
    if (!log_reader.get_fov().has_value()) {
        fov = 200;  
    } else {
        fov = *log_reader.get_fov();
    }
    bool flip_x_for_carla = false; 
    if (location.rfind("carla:", 0) == 0) {
        flip_x_for_carla = true;
    }
    LocationInfoRequest li_req("{}");
    li_req.set_location(location);
    li_req.set_include_map_source(true);
    li_req.set_rendering_fov(fov);
    li_req.set_rendering_center(log_reader.get_rendering_center());
    LocationInfoResponse li_res = location_info(li_req, &session);
    auto rc = log_reader.get_scenario_log().rendering_center;
    if (!rc) {
        std::cerr << "please provide a rendering center in JSON logs\n";
        return 1;
    }

    Visualizer viz(li_res, fov,*rc, flip_x_for_carla);
    viz.initialize_video("scenario_log_replay.avi", 10);
    log_reader.reset_log();
    do {
        const auto& states = log_reader.current_agent_states();
        const auto  props  = log_reader.current_agent_properties();
        auto traffic_lights_states = log_reader.current_traffic_lights();
        viz.render_step(states, props, traffic_lights_states);
    } while (log_reader.next());
    viz.close();

    // Choose an earlier timestep from which to branch off
    log_reader.reset_log();
    log_reader.initialize();  
    log_reader.return_state_at_timestep(TIMESTEP_TO_BRANCH_FROM);
    std::vector<AgentState> agent_states = log_reader.current_agent_states();
    std::vector<AgentProperties> agent_properties = log_reader.current_agent_properties();
    std::optional<std::map<std::string,std::string>> tl_states = log_reader.current_traffic_lights();
    std::optional<std::vector<LightRecurrentState>> light_rnn = log_reader.current_light_recurrent_state();
    std::optional<std::vector<RecurrentState>> rnn_opt = log_reader.current_recurrent_states();
    std::vector<std::vector<double>> api_rnn;
    if (rnn_opt.has_value()) {
        const auto& rnn_vec = *rnn_opt;
        for (const RecurrentState& rs : rnn_vec) {
            api_rnn.emplace_back(rs.packed.begin(), rs.packed.end());
        }
    }
    Visualizer viz_branched(li_res, fov, *rc, flip_x_for_carla);
    viz_branched.initialize_video("scenario_log_branched.avi", 10);

    for(int i = 0; i < NEW_ROLLOUT_LENGTH; i++) {
        DriveRequest drive_req("{}");
        drive_req.set_location(log_reader.get_location());
        drive_req.set_agent_states(agent_states);
        drive_req.set_agent_properties(agent_properties);
        drive_req.set_recurrent_states(api_rnn);
        if (light_rnn.has_value())
            drive_req.set_light_recurrent_states(*light_rnn);
        drive_req.set_rendering_center(log_reader.get_rendering_center());
        drive_req.set_rendering_fov(fov);

        DriveResponse resp = drive(drive_req, &session);
        agent_states = resp.agent_states();
        api_rnn    = resp.recurrent_states();
        tl_states    = resp.traffic_lights_states();
        light_rnn    = resp.light_recurrent_states();

        viz_branched.render_step(
            agent_states,
            agent_properties,
            tl_states
        );
    }
    viz_branched.close();

}