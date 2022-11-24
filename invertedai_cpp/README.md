# invertedai-cpp

## Development using docker

``` sh
docker compose build
docker compose run --rm dev
```


## Dependency
- gcc-10, g++-10
- [Bazel](https://bazel.build/install)
- Opencv (`sudo apt-get install -y libopencv-dev`)
- Boost (`sudo apt install openssl libssl-dev libboost-all-dev`)
- [JSON for Modern C++](https://json.nlohmann.me/) (under `invertedai/externals/json.hpp`)

## Usage
`bazel build //examples:client_example` to compile.
`./client $location $agent_num $timestep $api_key` to run.
e.g. `./bazel-bin/examples/client_example canada:vancouver:ubc_roundabout 5 20 "EFGHTOKENABCD"`

## Doc
`git clone` the repo, open the doc/index.html with a browser.

You can modify the Doxyfile and regenerate the doc by `doxygen Doxyfile`
