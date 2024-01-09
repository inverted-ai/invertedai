#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../invertedai/api.h"

using tcp = net::ip::tcp;    // from <boost/asio/ip/tcp.hpp>
using json = nlohmann::json; // from <json.hpp>

int main(int argc, char **argv) {
  // Set up drive scenarios from initialize
  const char* tests[] =
  {
    "examples/initialize_body.json",
    "examples/initialize_with_states_and_attributes.json",
    "examples/initialize_sampling_with_types.json"
  };

   try {
    const int timestep = std::stoi(argv[1]);
    const std::string api_key(argv[2]);

    net::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    // configure connection setting
    invertedai::Session session(ioc, ctx);
    session.set_api_key(api_key);
    session.connect();
    int i = 0;
    for (const char *test : tests) {
        invertedai::InitializeRequest init_req(invertedai::read_file(test));
        invertedai::InitializeResponse init_res = invertedai::initialize(init_req, &session);
        auto image = cv::imdecode(init_res.birdview(), cv::IMREAD_COLOR);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        int frame_width = image.rows;
        int frame_height = image.cols;
        std::string drive_video_name = "drive_test_" + std::to_string(i) + ".avi";
        cv::VideoWriter video(
            drive_video_name,
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
            10,
            cv::Size(frame_width, frame_height)
        );
        i += 1;
        invertedai::DriveRequest drive_req(invertedai::read_file("examples/drive_body.json"));
        drive_req.set_location(init_req.location());
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
    }
  } catch (std::exception const &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}