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
                ./bazel-bin/examples/scenario_log_example

*/
const int TIMESTEP_TO_BRANCH_FROM = 10;
const int NEW_ROLLOUT_LENGTH = 100;
int main(int argc, char** argv) {
    const std::string API_KEY = getenv("IAI_API_KEY"); 
    LogReader log_reader("examples/carla_Town10HD_log.json");
    boost::asio::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    invertedai::Session session(ioc, ctx);
    session.set_api_key(API_KEY);
    session.connect();

    const std::string location = log_reader.get_location();
    bool FLIP_X_FOR_THIS_DOMAIN = false; 
    if (location.rfind("carla:", 0) == 0) {
        FLIP_X_FOR_THIS_DOMAIN = true;
    }
    LocationInfoRequest li_req("{}");
    li_req.set_location(location);
    li_req.set_include_map_source(true);
    li_req.set_rendering_fov(log_reader.get_fov());
    li_req.set_rendering_center(log_reader.get_rendering_center());
    LocationInfoResponse li_res = location_info(li_req, &session);
    auto image = cv::imdecode(li_res.birdview_image(), cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    int frame_width  = image.cols;
    int frame_height = image.rows;
    
    cv::VideoWriter video(
        "scenario_log_replay.avi",
        cv::VideoWriter::fourcc('M','J','P','G'),
        10,  
        cv::Size(frame_width, frame_height)
    );

    auto rc = log_reader.get_scenario_log().rendering_center;
    if (!rc) {
        std::cerr << "please provide a rendering center in JSON logs\n";
        return 1;
    }

    double cx = rc->first;
    double cy = rc->second;
    double FOV = log_reader.get_fov();
    double half = FOV * 0.5;
    double min_x = cx - half;
    double max_y = cy + half;
    double scale = image.rows / FOV;
    WorldToPixelProjector world_to_pixel {
        .cx = cx,
        .cy = cy,
        .min_x = min_x,
        .max_y = max_y,
        .scale = scale,
        .flip_x = FLIP_X_FOR_THIS_DOMAIN,
        .width = frame_width,
        .height = frame_height
    };

    // Run through the entire log and render each timestep
    log_reader.reset_log();
    do {
        cv::Mat frame = image.clone();

        const auto& states = log_reader.current_agent_states();
        const auto  props  = log_reader.current_agent_properties();

        for (size_t i = 0; i < states.size(); ++i) {
            draw_agent(frame, states[i], props[i], world_to_pixel);
        }
        auto traffic_lights_states = log_reader.current_traffic_lights();
        if (traffic_lights_states.has_value()) {
            std::map<std::string, cv::Point> traffic_light_positions_px =
                get_traffic_light_positions(li_res.static_actors(), world_to_pixel);
            draw_traffic_lights(frame,
                traffic_lights_states,
                traffic_light_positions_px,
                li_res.static_actors(),
                world_to_pixel,
                FLIP_X_FOR_THIS_DOMAIN
            );
        }
        video.write(frame);

    } while (log_reader.next());

    video.release();

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
            api_rnn.emplace_back(
                rs.packed.begin(),
                rs.packed.end()
            );
        }
    }

    cv::VideoWriter video_branched(
        "scenario_log_branched.avi",
        cv::VideoWriter::fourcc('M','J','P','G'),
        10,  
        cv::Size(frame_width, frame_height)
    );

    for(int i = 0; i < NEW_ROLLOUT_LENGTH; i++) {
        DriveRequest drive_req("{}");
        drive_req.set_location(log_reader.get_location());
        drive_req.set_agent_states(agent_states);
        drive_req.set_agent_properties(agent_properties);
        drive_req.set_recurrent_states(api_rnn);
        if (light_rnn.has_value())
            drive_req.set_light_recurrent_states(*light_rnn);
        drive_req.set_rendering_center(log_reader.get_rendering_center());
        drive_req.set_rendering_fov(log_reader.get_fov());

        DriveResponse resp = drive(drive_req, &session);
        agent_states = resp.agent_states();
        api_rnn    = resp.recurrent_states();
        tl_states    = resp.traffic_lights_states();
        light_rnn    = resp.light_recurrent_states();

        cv::Mat frame_branched = image.clone();
        for (size_t k = 0; k < agent_states.size(); k++) {
            draw_agent(frame_branched, agent_states[k], agent_properties[k], world_to_pixel);
        }
        if (tl_states.has_value()) {
            std::map<std::string, cv::Point> traffic_light_positions_px =
                get_traffic_light_positions(li_res.static_actors(), world_to_pixel);
            draw_traffic_lights(
                frame_branched,
                tl_states,
                traffic_light_positions_px,
                li_res.static_actors(),
                world_to_pixel,
                FLIP_X_FOR_THIS_DOMAIN
            );
        }
        video_branched.write(frame_branched);
    }
    video_branched.release();

}