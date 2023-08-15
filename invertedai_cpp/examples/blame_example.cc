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

// usage: ./blame_example $api_key
int main(int argc, char **argv) {
  try {
    const std::string api_key(argv[1]);

    net::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    // configure connection setting
    invertedai::Session session(ioc, ctx);
    session.set_api_key(api_key);
    session.connect();

    // construct request for collision history
    invertedai::BlameRequest blame_req(
        invertedai::read_file("examples/blame_body.json"));

    std::cout << "blame request: \n" << blame_req.body_str() << std::endl;

    // get response of location information
    invertedai::BlameResponse blame_res =
        invertedai::blame(blame_req, &session);

    std::cout << "blame response: \n" << blame_res.body_str() << std::endl;

  } catch (std::exception const &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
