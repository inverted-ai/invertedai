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
CLI will override values from JSON files.

### To Run:  
`./client drive_json:<path/to/json> location_json:<path/to/json> init_json:<path/to/json> location:<iai_location_id> cars:<integer> timestep:<integer> apikey:<your_api_key>` 

### Command Line Arguments:  
#### 3 Optional JSON file paths:  
`drive_json:<path/to/json>` (optional)  
- Path to a JSON file with request defaults for DRIVE (e.g., examples/drive_body.json).
- If not provided, the client defaults to examples/drive_body.json.

`location_json:<path/to/json>` (optional)
- Path to a JSON file with request defaults for LOCATION INFO (e.g., examples/location_info_body.json).

- If not provided, the client defaults to examples/location_info_body.json.

`init_json:<path/to/json>` (optional)  
- Path to a JSON file with request defaults for INITIALIZE (e.g., examples/initialize_body.json).

- If not provided, the client defaults to examples/initialize_body.json.

#### Arguments that override the JSON file values:  
`location:<string>` (required unless location_json file contains it)  
- Simulation location in IAI format, e.g. iai:10th_and_dunbar or canada:vancouver:ubc_roundabout.
- Overrides the location value in the location_json json file

`timestep:<int>` (required unless init_json file contains it)  
- Number of simulation steps per iteration.
- Units are in 100ms (e.g., `timestep:50` would run the simulation for 5000ms or 5 seconds)

#### Argument to add additional agents into the simulation:  
`cars:<int>` (optional)  
- Number of additional car agents to spawn.
- Default value is 0

#### Required API Key  
`apikey:<your_api_key>` (required)  
- Your Inverted AI API key.  


### Example usages:  
Running with the default JSON files:  
`./bazel-bin/examples/client_example apikey:EFGHTOKENABCD`  

Running with all possible arguments:  
`./bazel-bin/examples/client_example init_json:examples/initialize_body.json drive_json:examples/drive_body.json location_json:examples/location_info_body.json location:iai:10th_and_dunbar cars:5 timestep:20 apikey:EFGHTOKENABCD`  

Running custom JSON files:  
`./bazel-bin/examples/client_example init_json:examples/my_own_init.json drive_json:examples/my_own_drive.json location_json:examples/my_own_loc.json apikey:EFGHTOKENABCD`  

Running with CLI to override default JSON file values:  
`./bazel-bin/examples/client_example location:canada:ubc_roundabout cars:5 timestep:20 apikey:EFGHTOKENABCD`  

Running default JSON files + 5 additional cars:  
`./bazel-bin/examples/blame_example cars:5 apikey:EFGHTOKENABCD`  

## Doc
`git clone` the repo, open the doc/index.html with a browser.  

You can modify the Doxyfile and regenerate the doc by `doxygen Doxyfile`
