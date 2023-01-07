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

// usage: ./client $location $agent_num $timestep $api_key
int main(int argc, char **argv) {
  try {
    const std::string location(argv[1]);
    const int agent_num = std::stoi(argv[2]);
    const int timestep = std::stoi(argv[3]);
    const std::string api_key(argv[4]);

    net::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    // configure connection setting
    invertedai::Session session(ioc, ctx);
    session.set_api_key(api_key);
    session.connect();

    // construct request for getting information about the location
    invertedai::LocationInfoRequest loc_info_req(
        "{\"location\": \"" + location +
        "\", "
        "\"include_map_source\": true}");
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

    // construct request for initializing the simulation (placing NPCs on the
    // map)
    invertedai::InitializeRequest init_req(
        invertedai::read_file("examples/initialize_body.json"));
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

    for (int t = 0; t < timestep; t++) {
      // step the simulation by driving the agents
      invertedai::DriveResponse drive_res =
          invertedai::drive(drive_req, &session);

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
