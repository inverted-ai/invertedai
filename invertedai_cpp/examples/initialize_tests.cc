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
  // list of test scenarios to run
  const char* tests[] =
  {
    "examples/initialize_body.json",
    "examples/initialize_with_states_and_attributes.json",
    "examples/initialize_sampling_with_types.json"
  };

  try {
    const std::string api_key(argv[1]);

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
        std::string img_name = "initialize_test_" + std::to_string(i) + ".png";
        cv::imwrite(img_name, image);
        i += 1;
    }
  } catch (std::exception const &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}