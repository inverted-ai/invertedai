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

    // get response of location information
    invertedai::BlameResponse blame_res =
        invertedai::blame(blame_req, &session);

    //Confirm the returned blame response
    std::vector<int> agents_at_fault = blame_res.agents_at_fault();
    std::cout << "Agents at fault: [";
    for (auto & agent_num : agents_at_fault) {
      std::cout << agent_num << ", ";
    }
    std::cout << "]" << std::endl;

    std::optional<std::map<int, std::vector<std::string>>> reasons = blame_res.reasons();
    if (reasons) { //Check whether a reasons map was returned
      std::cout << "Reasons for faulted agents: [";

      for (const auto &agent : reasons.value()){
        std::cout << "[Reason for agent number " << agent.first << ": ";
          for (auto & reason : agent.second) {
            std::cout << reason << ", ";
          }
          std::cout << "], ";
      } 
      std::cout << "]" << std::endl;
    }
    else {
      std::cout << "Reasons disabled in this response." << std::endl;
    }
    

  } catch (std::exception const &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
