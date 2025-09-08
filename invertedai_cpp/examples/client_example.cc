#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../invertedai/api.h"

using tcp = net::ip::tcp;    // from <boost/asio/ip/tcp.hpp>
using json = nlohmann::json; // from <json.hpp>

// command line arguments:
// if JSON file is provided, location and timestep are optional
// users can call any of the following arguments in any order
// example usage: ./bazel-bin/examples/client_example json:examples/initialize_body.json location:iai:10th_and_dunbar cars:5 timestep:20 apikey:xxxxxx
// example usage: ./client json:examples/initialize_body.json apikey:xxxxxx
// example usage: ./client location:iai:ubc_roundabout cars:5 timestep:20 apikey:xxxxxx
int main(int argc, char **argv) {
  try {
    std::optional<std::string> json_file;
    std::string location;
    int car_agent_num = 0;
    int timestep = -1;
    std::string api_key;
    for(int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg.find("cars:") == 0) car_agent_num = std::stoi(arg.substr(5));
      else if (arg.find("timestep:") == 0) timestep = std::stoi(arg.substr(9));
      else if (arg.find("apikey:") == 0) api_key = arg.substr(7);
      else if (arg.find("location:") == 0) location = arg.substr(9);
      else if (arg.find("json:") == 0) json_file = arg.substr(5);
      else {
        std::cerr << "Unknown argument: " << arg << std::endl;
        return EXIT_FAILURE;
      }
    }
    // mandatory API argument check
    if(api_key.empty()) {
      std::cerr << "API key is required. Please provide it using the 'apikey:' argument." << std::endl;
      return EXIT_FAILURE;
    }

    net::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    // configure connection setting
    invertedai::Session session(ioc, ctx);
    session.set_api_key(api_key);
    session.connect();

    // construct request for getting information about the location and timestep
    std::optional<invertedai::LocationInfoResponse> loc_info_res;
    // if a JSON file is provided in command line, load location and timestep from it if available
    if (json_file.has_value()) {
      invertedai::LocationInfoRequest loc_info_req(invertedai::read_file(json_file.value().c_str()));
      // use location from JSON if exists, otherwise use CLI location
      if (loc_info_req.location().has_value()) {
        location = loc_info_req.location().value();  
      } else if (location != "") {
          loc_info_req.set_location(location); 
      } else {
          std::cerr << "Error: No location specified in JSON or CLI." << std::endl;
          return EXIT_FAILURE;
      }
      // use timestep from JSON if exists
      if (loc_info_req.timestep().has_value()) {
          timestep = loc_info_req.timestep().value(); 
      }
      // set the location in the request
      loc_info_req.set_location(location);
      std::string body = loc_info_req.body_str();
      // get response of location information
      loc_info_res = invertedai::location_info(loc_info_req, &session);
    } else {
      // if no location and timestep provided, exit with error
      if (location.empty() || timestep <= 0) {
        std::cerr << "Error: location and timestep must be set via JSON or CLI." << std::endl;
        return EXIT_FAILURE;
      }
      // if no JSON file is provided, create a request with location from CLI
      invertedai::LocationInfoRequest loc_info_req("{}");
      loc_info_req.set_location(location);
      loc_info_res = invertedai::location_info(loc_info_req, &session);
    }

    // use opencv to decode and save the bird's eye view image of the simulation
    auto image = cv::imdecode(loc_info_res->birdview_image(), cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    int frame_width = image.rows;
    int frame_height = image.cols;
    cv::VideoWriter video(
      "iai-demo.avi",
      cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 
      10,
      cv::Size(frame_width, frame_height)
    );
    // construct request for initializing the simulation (placing NPCs on the
    // map)
    invertedai::InitializeRequest init_req(
      json_file.has_value() ? invertedai::read_file(json_file.value().c_str()) : "{}"
    );
    // set the location
    init_req.set_location(location);
    // spawn agents in simulation
    std::vector<invertedai::AgentProperties> agent_properties;
    // load initial agent properties from the JSON file if any
    if (init_req.agent_properties().has_value()) {
      agent_properties = init_req.agent_properties().value();  
  }
  // add car_agent_num number of cars on top of the JSON file agents
    for(int i = 0; i < car_agent_num; i++) {
      invertedai::AgentProperties ap;
      ap.agent_type = std::string("car");
      agent_properties.push_back(ap);
    }
    // apply the agent properties to the initialization request
    init_req.set_agent_properties(agent_properties);
    // set the number of agents 
    init_req.set_num_agents_to_spawn(agent_properties.size());
    // get the response of simulation initialization
    invertedai::InitializeResponse init_res = invertedai::initialize(init_req, &session);

    // construct request for stepping the simulation (driving the NPCs)
    invertedai::DriveRequest drive_req(invertedai::read_file("examples/drive_body.json"));
    drive_req.set_location(location);
    drive_req.update(init_res);

    for (int t = 0; t < timestep; t++) {
      // step the simulation by driving the agents
      invertedai::DriveResponse drive_res = invertedai::drive(drive_req, &session);
      // use opencv to decode and save the bird's eye view image of the
      // simulation
      auto image = cv::imdecode(drive_res.birdview(), cv::IMREAD_COLOR);
      cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
      video.write(image);
      drive_req.update(drive_res);
      std::cout << "Remaining iterations: " << timestep - t << std::endl;
    }
    video.release();
  } catch (std::exception const &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
