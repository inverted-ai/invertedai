#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <future> //includes std::async library
#include <chrono>
#include <stdexcept>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../invertedai/api.h"
#include "../invertedai/data_utils.h"

using tcp = net::ip::tcp;    // from <boost/asio/ip/tcp.hpp>
using json = nlohmann::json; // from <json.hpp>

std::vector<std::vector<invertedai::AgentState>> linear_interpolation(std::vector<invertedai::AgentState> current_agent_states, std::vector<invertedai::AgentState> next_agent_states, int number_steps) {
  std::vector<std::vector<invertedai::AgentState>> interpolated_states;
  for (int tt = 0; tt < number_steps; tt++) {
    std::vector<invertedai::AgentState> timestep_states;
    for(int i = 0; i < (int)next_agent_states.size(); i++){
      invertedai::AgentState agent_state;
      agent_state.x = current_agent_states[i].x + (next_agent_states[i].x - current_agent_states[i].x)*(tt/number_steps);
      agent_state.y = current_agent_states[i].y + (next_agent_states[i].y - current_agent_states[i].y)*(tt/number_steps);
      agent_state.orientation = current_agent_states[i].orientation + (next_agent_states[i].orientation - current_agent_states[i].orientation)*(tt/number_steps);
      agent_state.speed = current_agent_states[i].speed + (next_agent_states[i].speed - current_agent_states[i].speed)*(tt/number_steps);
      timestep_states.push_back(agent_state);
    }
    interpolated_states.push_back(timestep_states);
  }

  return interpolated_states;
}

// usage: ./client $location $agent_num $timestep $api_key
int main(int argc, char **argv) {
  try {
    const std::string location(argv[1]);
    const int agent_num = std::stoi(argv[2]);
    const int timestep = std::stoi(argv[3]);
    const std::string api_key(argv[4]);
    const int FPS = 30;
    const int IAI_FPS = 10;
    if (FPS % IAI_FPS != 0) {
      throw std::invalid_argument("FPS argument must be an integer multiple of 10.");
    }
    int NUM_INTERP_STEPS = (int) FPS/IAI_FPS;

    net::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    // configure connection setting
    invertedai::Session session(ioc, ctx);
    session.set_api_key(api_key);
    session.connect();

    // construct request for getting information about the location
    invertedai::LocationInfoRequest loc_info_req(
        invertedai::read_file("examples/location_info_body.json"));
    loc_info_req.set_location(location);

    // get response of location information
    invertedai::LocationInfoResponse loc_info_res =
        invertedai::location_info(loc_info_req, &session);

    // use opencv to decode and save the bird's eye view image of the simulation
    auto image = cv::imdecode(loc_info_res.birdview_image(), cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    int frame_width = image.rows;
    int frame_height = image.cols;
    cv::VideoWriter video("iai-demo.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10,
                          cv::Size(frame_width, frame_height));

    invertedai::AgentState ego_agent_state;
    ego_agent_state.x = -28.32;
    ego_agent_state.y = 47.17;
    ego_agent_state.orientation = -1.0;
    ego_agent_state.speed = 3.0;

    invertedai::AgentAttributes ego_agent_attribute;
    ego_agent_attribute.length = 4.77;
    ego_agent_attribute.width = 1.53;
    ego_agent_attribute.rear_axis_offset = 1.51;

    json init_parameters;
    init_parameters["location"] = "iai:ubc_roundabout";
    init_parameters["num_agents_to_spawn"] = 10;
    init_parameters["states_history"] = NULL;
    init_parameters["agent_attributes"] = NULL;
    init_parameters["traffic_light_state_history"] = NULL;
    init_parameters["get_birdview"] = false;
    init_parameters["get_infractions"] = false;
    init_parameters["random_seed"] = NULL;
    init_parameters["location_of_interest"] = NULL;
    init_parameters["conditional_agent_states"] = {
      ego_agent_state.x, 
      ego_agent_state.y,
      ego_agent_state.orientation,
      ego_agent_state.speed
    };
    init_parameters["conditional_agent_attributes"] = {
      ego_agent_attribute.length, 
      ego_agent_attribute.width,
      ego_agent_attribute.rear_axis_offset
    };
    std::string init_parameters_str = init_parameters.dump();
    /*
    std::string init_parameters = R"(
      {
        "location": "iai:ubc_roundabout",
        "num_agents_to_spawn": 10,
        "states_history": null,
        "agent_attributes": null,
        "traffic_light_state_history": null,
        "get_birdview": false,
        "get_infractions": false,
        "random_seed": null,
        "location_of_interest": null,
        "conditional_agent_states": [)" +
        to_string(ego_agent.x) + R"(,)" +
        to_string(ego_agent.y) + R"(,)" +
        to_string(ego_agent.orientation) + R"(,)" +
        to_string(ego_agent.speed) +
        R"(
        ],
        "conditional_agent_attributes": [)" +
        to_string(ego_agent_attribute.length) + R"(,)" +
        to_string(ego_agent_attribute.width) + R"(,)" +
        to_string(ego_agent_attribute.rear_axis_offset) +
        R"(
        ]
      }
    )";
    */

    // construct request for initializing the simulation (placing NPCs on the
    // map)
    //invertedai::InitializeRequest init_req(invertedai::read_file("examples/initialize_body.json"));
    invertedai::InitializeRequest init_req(init_parameters_str);
    // set the location
    init_req.set_location(location);
    // set the number of agents
    init_req.set_num_agents_to_spawn(agent_num);
    // get the response of simulation initialization
    invertedai::InitializeResponse init_res =
        invertedai::initialize(init_req, &session);

    // construct request for stepping the simulation (driving the NPCs)
    invertedai::DriveRequest drive_req(
        invertedai::read_file("examples/drive_body.json"));
    drive_req.set_location(location);
    drive_req.update(init_res);
    
    invertedai::DriveResponse next_drive_res = invertedai::drive(drive_req, &session);
    std::vector<invertedai::AgentState> next_agent_states = next_drive_res.agent_states();
    std::vector<invertedai::AgentState> current_agent_states = init_res.agent_states();
    drive_req.update(next_drive_res);

    for (int t = 0; t < timestep; t++) {
      // step the simulation by driving the agents
      std::future<invertedai::DriveResponse> drive_res = std::async (std::launch::async,&invertedai::drive,drive_req,&session);
      
      std::vector<std::vector<invertedai::AgentState>> interpolated_states = linear_interpolation(current_agent_states,next_agent_states,NUM_INTERP_STEPS);
      
      next_drive_res = drive_res.get();
      next_agent_states = next_drive_res.agent_states();
      current_agent_states = next_agent_states;

      auto image = cv::imdecode(next_drive_res.birdview(), cv::IMREAD_COLOR);
      cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
      video.write(image);
      drive_req.update(next_drive_res);
      std::cout << "Remaining iterations: " << timestep - t << std::endl;
    }
    video.release();
  } catch (std::exception const &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
