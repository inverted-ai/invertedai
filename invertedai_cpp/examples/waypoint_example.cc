
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../invertedai/api.h"
#include "../invertedai/data_utils.h"

using AgentAttributes = invertedai::AgentAttributes;
using Point2d = invertedai::Point2d;


int processScenario(const char *bodyPath, const std::string api_key, int timestep);

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <timestep> <api_key>\n";
        return EXIT_FAILURE;
    }

    int timestep;
    try {
        timestep = std::stoi(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Invalid timestep argument: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    const std::string api_key(argv[2]);
    const char* example_body = "examples/initialize_body_waypoint_car_example.json";

    try {
          if (processScenario(example_body, api_key, timestep) != EXIT_SUCCESS) {
              return EXIT_FAILURE;
          }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int processScenario(const char *bodyPath, const std::string api_key, int timestep) {
    using tcp = net::ip::tcp;    // from <boost/asio/ip/tcp.hpp>
    using json = nlohmann::json; // from <json.hpp>

    net::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    invertedai::Session session(ioc, ctx);
    session.set_api_key(api_key);
    session.connect();

    invertedai::InitializeRequest init_req(invertedai::read_file(bodyPath));
    invertedai::InitializeResponse init_res = invertedai::initialize(init_req, &session);
    auto image = cv::imdecode(init_res.birdview(), cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    int frame_width = image.rows;
    int frame_height = image.cols;
    std::string video_name = "waypoint_example.mp4" ;

    cv::VideoWriter video(
        video_name,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        10,
        cv::Size(frame_width, frame_height)
    );

    invertedai::DriveRequest drive_req(invertedai::read_file("examples/drive_body.json"));
    drive_req.set_location(init_req.location());
    drive_req.update(init_res);
    drive_req.set_rendering_center(std::make_pair(313, -194)); // Render the optional birdview in a reasnable area
    drive_req.set_rendering_fov(300);
    std::vector<AgentAttributes> agent_attributes = drive_req.agent_attributes();

    for (int t = 0; t < timestep; t++) {
        invertedai::DriveResponse drive_res = invertedai::drive(drive_req, &session);
        int agent_idx = 0;
        for (AgentAttributes attr : agent_attributes){
          if (attr.waypoint.has_value() &&  attr.waypoint.value().isCloseTo({drive_res.agent_states()[agent_idx].x, drive_res.agent_states()[agent_idx].y}, 2))  {
            attr.waypoint = std::nullopt;
            drive_req.update_attribute(agent_idx, attr);
            std::cout << "Agent " << agent_idx << " reached waypoint" << std::endl;
          }
          agent_idx ++;
        }
        auto image = cv::imdecode(drive_res.birdview(), cv::IMREAD_COLOR);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        video.write(image);
        drive_req.update(drive_res);
        std::cout << "Remaining iterations: " << timestep - t << std::endl;
    }

    return EXIT_SUCCESS;
}
