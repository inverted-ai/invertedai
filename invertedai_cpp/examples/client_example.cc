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

// all configuration options can be set via command line arguments
struct CommandLineArgs {
  std::string api_key;
  std::optional<std::string> location;
  int car_agent_num = 0;
  std::optional<int> timestep;

  std::string location_json_path = "examples/location_info_body.json"; // default file paths
  std::string init_json_path     = "examples/initialize_body.json";    // placed here for ease of access
  std::string drive_json_path    = "examples/drive_body.json";         
};

static CommandLineArgs parse_args(int argc, char** argv) {
  CommandLineArgs cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg.rfind("apikey:", 0) == 0) {
      cfg.api_key = arg.substr(7);
    } else if (arg.rfind("location:", 0) == 0) {
      cfg.location = arg.substr(9);
    } else if (arg.rfind("cars:", 0) == 0) {
      cfg.car_agent_num = std::stoi(arg.substr(5));
    } else if (arg.rfind("timestep:", 0) == 0) {
      cfg.timestep = std::stoi(arg.substr(9));
    } else if (arg.rfind("location_json:", 0) == 0) {
      cfg.location_json_path = arg.substr(14);
    } else if (arg.rfind("init_json:", 0) == 0) {
      cfg.init_json_path = arg.substr(10);
    } else if (arg.rfind("drive_json:", 0) == 0) {
      cfg.drive_json_path = arg.substr(11);
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  return cfg;
}

static invertedai::LocationInfoResponse
send_location_info(const CommandLineArgs& cfg,
                   invertedai::Session* session,
                   std::string& out_final_location) {
  std::string loc_body = "{}";
  try {
    loc_body = invertedai::read_file(cfg.location_json_path.c_str());
  } catch (const std::exception& e) {
    std::cerr << "Warning: Could not read location info JSON file: " << e.what() << std::endl;
  }

  invertedai::LocationInfoRequest loc_info_req(loc_body);

  // CLI overrides JSON if provided
  if (cfg.location && !cfg.location->empty()) {
    loc_info_req.set_location(*cfg.location);
  }

  // Final location must be present in the request before calling the API
  const std::string loc_req_location = loc_info_req.location().value();
  if (loc_req_location.empty()) {
    std::cerr << "Error: location must be provided via CLI or location JSON." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  out_final_location = loc_req_location;

  // call API
  auto res = invertedai::location_info(loc_info_req, session);
  std::cout << "Location info received for location: " << loc_req_location << std::endl;
  return res;
}

static invertedai::InitializeResponse do_initialize(const CommandLineArgs& cfg, invertedai::Session* session, std::string& location) {
  std::string init_body = "{}";
  try {
    init_body = invertedai::read_file(cfg.init_json_path.c_str());
    // get time step from JSON initialization file if not provided in CLI
  } catch (const std::exception& e) {
    std::cerr << "Warning: Could not read initialize JSON file: " << e.what() << std::endl;
  }

  invertedai::InitializeRequest init_req(init_body);
  // make sure correct location is set
  init_req.set_location(location);

  // add car agents
  std::vector<invertedai::AgentProperties> agent_properties;
  if (init_req.agent_properties().has_value()) {
    agent_properties = init_req.agent_properties().value();
  }
  for (int i = 0; i < cfg.car_agent_num; i++) {
    invertedai::AgentProperties ap;
    ap.agent_type = "car";
    agent_properties.push_back(ap);
  }
  init_req.set_agent_properties(agent_properties);
  init_req.set_num_agents_to_spawn(agent_properties.size());

  // call API
  auto res = invertedai::initialize(init_req, session);
  std::cout << "Simulation initialized with " << agent_properties.size() << " agents." << std::endl;
  return res;

}

static invertedai::DriveRequest make_drive_request(const CommandLineArgs& cfg, const invertedai::InitializeResponse& init_res, std::string location) {
  std::string drive_body = "{}";
  try {
    drive_body = invertedai::read_file(cfg.drive_json_path.c_str());
  } catch (const std::exception& e) {
    std::cerr << "Warning: Could not read drive JSON file: " << e.what() << std::endl;
  }

  invertedai::DriveRequest drive_req(drive_body);
  drive_req.set_location(location);
  drive_req.update(init_res);

  return drive_req;
}

static int getNumericValueFromJson(const std::string& json_str, const std::string& key) {
  try {
    std::string body = invertedai::read_file(json_str.c_str());
    auto j = nlohmann::json::parse(body);
    if (j.contains(key)) {
      return j[key];
    }
  } catch (...) {
    // ignore
  }
  return 0;
}

static std::string getStringValueFromJson(const std::string& json_str, const std::string& key) {
  try {
    std::string body = invertedai::read_file(json_str.c_str());
    auto j = nlohmann::json::parse(body);
    if (j.contains(key)) {
      return j[key];
    }
  } catch (...) {
    // ignore
  }
  return "";
}

static void validate(const CommandLineArgs& cfg) {
  // check mandatory API key
  if (cfg.api_key.empty()) {
    std::cerr << "API key is required (apikey:<key>)." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // if timestep is not provided in CLI, it must be present in the initialization JSON
  if (!cfg.timestep.has_value()) {
    int init_timestep = getNumericValueFromJson(cfg.init_json_path, "timestep");
    if (init_timestep == 0) {
      std::cerr << "Error: timestep must be set via JSON or CLI." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // location must be provided in CLI or location JSON
  if (!cfg.location.has_value() || cfg.location->empty()) {
    std::string loc_in_json = getStringValueFromJson(cfg.location_json_path, "location");
    if (loc_in_json.empty()) {
      std::cerr << "Error: location must be provided via CLI or LocationInfo JSON." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // locations should be consistent across all JSON files if provided and CLI location not provided
  if(cfg.location.has_value() && !cfg.location->empty()) {
    return; // CLI location takes precedence, no need to check consistency
  } else if(getStringValueFromJson(cfg.init_json_path, "location") == "") {
    std::cerr << "Warning: location not provided in Initialize JSON" << std::endl;
  }else if(getStringValueFromJson(cfg.init_json_path, "location") != getStringValueFromJson(cfg.location_json_path, "location")) {
    std::cerr << "Warning: location in Initialize JSON does not match LocationInfo location." << std::endl;
  } else if(getStringValueFromJson(cfg.drive_json_path, "location") == "") {
    std::cerr << "Error: location must be provided in Drive JSON" << std::endl;
  }else if(getStringValueFromJson(cfg.drive_json_path, "location") != ""
   && getStringValueFromJson(cfg.drive_json_path, "location") != getStringValueFromJson(cfg.location_json_path, "location")) {
    std::cerr << "Warning: location in Drive JSON does not match LocationInfo location." << std::endl;
  }
}

// See the README.md for invertedai_cpp for instructions on how to run the executable and mandatory/optional arguments
// example usage: ./bazel-bin/examples/client_example drive_json:examples/drive_body.json location_json:examples/location_info_body.json init_json:examples/initialize_body.json location:iai:10th_and_dunbar cars:5 timestep:20 apikey:xxxxxx
int main(int argc, char **argv) {
  try {
    // parse arguments from CLI
    CommandLineArgs cfg = parse_args(argc, argv);

    // validate configuration
    validate(cfg);

    // session configuration
    net::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    // configure connection setting
    invertedai::Session session(ioc, ctx);
    session.set_api_key(cfg.api_key);
    session.connect();

    // LocationInfo
    // created local variable to hold the final location used in all requests
    std::string final_location;
    auto loc_info_res = send_location_info(cfg, &session, final_location);

    // use opencv to decode and save the bird's eye view image of the simulation
    auto image = cv::imdecode(loc_info_res.birdview_image(), cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    int frame_width = image.rows;
    int frame_height = image.cols;
    cv::VideoWriter video(
      "iai-demo.avi",
      cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 
      10,
      cv::Size(frame_width, frame_height)
    );

    // initialization
    auto init_res = do_initialize(cfg, &session, final_location);

    // obtain final timestep from JSON or CLI
    int final_timestep;
    // if timestep not provided in CLI, try to get it from JSON
    if(!cfg.timestep.has_value()) {
      final_timestep = getNumericValueFromJson(cfg.init_json_path, "timestep");
    } else {
      final_timestep = cfg.timestep.value();
    }

    // construct request for stepping the simulation (driving the NPCs)
    auto drive_req = make_drive_request(cfg, init_res, final_location);

    for (int t = 0; t < final_timestep; t++) {
      // step the simulation by driving the agents
      invertedai::DriveResponse drive_res = invertedai::drive(drive_req, &session);
      // use opencv to decode and save the bird's eye view image of the
      // simulation
      auto image = cv::imdecode(drive_res.birdview(), cv::IMREAD_COLOR);
      cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
      video.write(image);
      drive_req.update(drive_res);
      std::cout << "Remaining iterations: " << final_timestep - t << std::endl;
    }
    video.release();
  } catch (std::exception const &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
