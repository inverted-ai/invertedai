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
- [Rules Boost](https://github.com/nelhage/rules_boost) (Please update the [latest commit](https://github.com/nelhage/rules_boost/commits/master) in the `WORKSPACE`)
- [JSON for Modern C++](https://json.nlohmann.me/) (under `invertedai/externals/json.hpp`)

## Usage
`bazel build //examples:client_example` to compile.  
The client executable accepts arguments in the form key:value.  
Arguments can be provided in any order.  
Some arguments are optional depending on values provided in the JSON file.  
JSON file values will always override the CLI.  

#### To Run:  
`./client json:<path/to/json> location:<iai_location_id> cars:<integer> timestep:<integer> apikey:<your_api_key>` 

#### Arguments:  
json:<file> (optional)  
- Path to a JSON file with request defaults (e.g., examples/initialize_body.json).
- If not provided, the client will start with an empty request and you must set values via CLI.

location:<string> (required unless JSON contains it)  
- Simulation location in IAI format, e.g. iai:10th_and_dunbar or canada:vancouver:ubc_roundabout.

timestep:<int> (required unless JSON contains it)  
- Number of simulation steps per iteration.

cars:<int> (optional if JSON file contains agent_attributes)  
- Number of additional car agents to spawn.
- Default value is 0

apikey:<string> (required)  
- Your Inverted AI API key.


#### Example usages:  
Running with all possible arguments:  
`./bazel-bin/examples/client_example json:examples/initialize_body.json location:iai:10th_and_dunbar cars:5 timestep:20 apikey:EFGHTOKENABCD`  

Running without a JSON file:  
`./bazel-bin/examples/client_example location:canada:ubc_roundabout cars:5 timestep:20 apikey:EFGHTOKENABCD` 

Running with only a JSON file:  
`./bazel-bin/examples/blame_example json:examples/initialize_body.json apikey:EFGHTOKENABCD`  

Running with a JSON file + 5 additional cars:  
`./bazel-bin/examples/blame_example json:examples/initialize_body.json cars:5 apikey:EFGHTOKENABCD`  

## Doc
`git clone` the repo, open the doc/index.html with a browser.

You can modify the Doxyfile and regenerate the doc by `doxygen Doxyfile`
