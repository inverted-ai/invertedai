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

    net::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    invertedai::Session session(ioc, ctx);
    session.connect();
    session.set_api_key(argv[4]);

    invertedai::LocationInfoRequest loc_info_req("{\"location\": \"" + location +
                                     "\", "
                                     "\"include_map_source\": true}");
    invertedai::LocationInfoResponse loc_info_res = invertedai::location_info(loc_info_req, &session);

    auto image = cv::imdecode(loc_info_res.birdview_image(), cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    int frame_width = image.rows;
    int frame_height = image.cols;
    cv::VideoWriter video("iai-demo.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10,
                          cv::Size(frame_width, frame_height));

    invertedai::InitializeRequest init_req(invertedai::read_file("examples/initialize_body.json"));
    init_req.set_location(location);
    init_req.set_num_agents_to_spawn(agent_num);
    invertedai::InitializeResponse init_res = invertedai::initialize(init_req, &session);

    invertedai::DriveRequest drive_req(invertedai::read_file("examples/drive_body.json"));
    drive_req.update(init_res);

    for (int t = 0; t < timestep; t++) {
      invertedai::DriveResponse drive_res = invertedai::drive(drive_req, &session);
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
