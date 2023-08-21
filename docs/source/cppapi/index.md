# C++ SDK

This page is a detailed reference for the C++ library. The underlying [REST API](https://app.swaggerhub.com/apis-docs/swaggerhub59/Inverted-AI/0.0.2) can also be
accessed directly. Below are the key functions of the library, along with some common utilities.


```{toctree}
:maxdepth: 1

cpp-blame
cpp-drive
cpp-initialize
cpp-location-info
cpp-common
```


## Quick Start with C++ SDK
### Docker
Clone the [Inverted AI repository](https://github.com/inverted-ai/invertedai.git), or just download the [CPP Library](https://download-directory.github.io/?url=https://github.com/inverted-ai/invertedai/tree/master/invertedai_cpp). \n
Make sure Docker is installed and running.
This can be done by running `docker info` in the terminal. \n
Navigate to the library directory (`cd invertedai_cppA`) and run the following commands in the terminal:
``` sh
docker compose build
docker compose run --rm dev
```

### Running a demo
- First, complie with: `bazel build //examples:client_example`
- Then, run: `./client_example $location $agent_num $timestep $api_key`,
e.g. `./bazel-bin/examples/client_example canada:vancouver:ubc_roundabout 5 20 xxxxxx`.


### Simple example

``` c++
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "../invertedai/api.h"

using tcp = net::ip::tcp;    // from <boost/asio/ip/tcp.hpp>
using json = nlohmann::json; // from <json.hpp>

// usage: ./client $location $agent_num $timestep $api_key
int main(int argc, char **argv) {
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

    // construct request for initializing the simulation (placing NPCs on the map)
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
    drive_req.update(init_res);

    for (int t = 0; t < timestep; t++) {
      // step the simulation by driving the agents
      invertedai::DriveResponse drive_res =
          invertedai::drive(drive_req, &session);
    }
  return EXIT_SUCCESS;
}
```
### Manual installation
Install the following dependencies:
- GCC-10 and g++-10
- [Bazel](https://bazel.build/install)
- Opencv (`sudo apt-get install -y libopencv-dev`)
- Boost (`sudo apt install openssl libssl-dev libboost-all-dev`)
- [JSON for Modern C++](https://json.nlohmann.me/) (under `invertedai/externals/json.hpp`)






