#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "invertedai/api.h"
#include "invertedai/data_utils.h"
#include "invertedai/drive_request.h"
#include "invertedai/drive_response.h"
#include "invertedai/initialize_request.h"
#include "invertedai/initialize_response.h"
#include "invertedai/location_info_request.h"
#include "invertedai/location_info_response.h"
#include "invertedai/logger.h"
#include "invertedai/session.h"
#include "invertedai/visualize.h"
#include "large/large_drive/large_drive.h"
#include "large/large_initialize/large_initialize.h"
#include "large/large_initialize/large_init_helpers.h"

using namespace invertedai;

/*
            HOW TO RUN EXECUTABLE:

            1. cd into invertedai_cpp folder

            2. Join docker:
                docker compose build
                docker compose run --rm dev

            3. Export your API key in the docker:
                export IAI_API_KEY="your_key_here"

            4. Build:
                bazel build //examples:scenario_replay_ego_takeover_example

            5. Run with defaults (replays examples/carla_Town10HD_log.json):
                ./bazel-bin/examples/scenario_replay_ego_takeover_example

            6. Run with overrides:
                ./bazel-bin/examples/scenario_replay_ego_takeover_example \
                    --log-path examples/carla_Town10HD_log.json \
                    --num-agents 5 --sim-length 50 --takeover-timestep 10

    This example is a C++ port of examples/scenario_replay_ego_takeover_example.py.
    Log agents are replayed from a JSON scenario log on indices [0, NUM_LOG_AGENTS),
    while sampled NPCs spawned around them are driven by the API. Once ts reaches
    --takeover-timestep, the ego (index --ego-id) is no longer forced from the log
    and instead follows the API-produced state, simulating an ego "takeover".
*/

namespace {

// reinit_cfg.regions must not request NPC sampling, otherwise large_initialize
// will spawn extra agents on top of the caller's intended population.
void update_agent_population(
    std::vector<AgentState> &agent_states,
    std::vector<AgentProperties> &agent_properties,
    std::vector<std::vector<double>> &recurrent_states,
    LargeInitializeConfig &reinit_cfg,
    const std::vector<std::size_t> &agent_ids_to_remove = {},
    const std::vector<AgentState> &agent_states_to_add = {},
    const std::vector<AgentProperties> &agent_properties_to_add = {}
) {
    if (agent_states_to_add.size() != agent_properties_to_add.size()) {
        throw std::invalid_argument(
            "agent_states_to_add and agent_properties_to_add must be the same length"
        );
    }
    if (agent_ids_to_remove.empty() && agent_states_to_add.empty()) {
        return;
    }

    // Erase from the highest index down so earlier indices stay valid.
    std::vector<std::size_t> sorted_ids = agent_ids_to_remove;
    std::sort(
        sorted_ids.begin(),
        sorted_ids.end(),
        std::greater<std::size_t>()
    );
    auto unique_end = std::unique(
        sorted_ids.begin(),
        sorted_ids.end()
    );
    sorted_ids.erase(
        unique_end,
        sorted_ids.end()
    );
    for (std::size_t id : sorted_ids) {
        if (id >= agent_states.size()) {
            throw std::out_of_range(
                "agent_id_to_remove " + std::to_string(id) + " is out of range"
            );
        }
        agent_states.erase(agent_states.begin() + id);
        agent_properties.erase(agent_properties.begin() + id);
        recurrent_states.erase(recurrent_states.begin() + id);
    }

    agent_states.insert(
        agent_states.end(),
        agent_states_to_add.begin(),
        agent_states_to_add.end()
    );
    agent_properties.insert(
        agent_properties.end(),
        agent_properties_to_add.begin(),
        agent_properties_to_add.end()
    );

    reinit_cfg.agent_states = agent_states;
    reinit_cfg.agent_properties = agent_properties;
    InitializeResponse response = large_initialize(reinit_cfg);
    agent_states = response.agent_states();
    agent_properties = response.agent_properties();
    recurrent_states = response.recurrent_states();
}

void print_usage(const char *bin) {
    std::cout << "Usage: " << bin << " [options]\n\n"
              << "Options:\n"
              << "  --log-path <str>             Path to scenario log JSON (default: examples/carla_Town10HD_log.json)\n"
              << "  --num-agents <int>           Number of background NPCs to spawn (default: 1)\n"
              << "  --sim-length <int>           Simulation length in timesteps (default: 100)\n"
              << "  --takeover-timestep <int>    Timestep at which the ego switches from log to DRIVE (default: 1)\n"
              << "  --ego-id <int>               Index of the ego inside the log agents list (default: 0)\n"
              << "  --fov <int>                  Rendering field of view in meters (default: 100)\n"
              << "  --width <int>                Width of NPC initialization area in meters (default: 100)\n"
              << "  --height <int>               Height of NPC initialization area in meters (default: 100)\n"
              << "  --scenario-center-x <float>  Optional scenario center x; falls back to map_origin if either x or y is unset\n"
              << "  --scenario-center-y <float>  Optional scenario center y; falls back to map_origin if either x or y is unset\n"
              << "  --get-infractions            Capture infraction data during simulation (default: off)\n"
              << "  --model-version-drive <str>  DRIVE model version (default: empty == latest)\n"
              << "  --help, -h                   Print this message\n";
}

} // namespace

int main(
    int argc,
    char **argv
) {
    std::string log_path = "examples/carla_Town10HD_log.json";
    int num_agents = 1;
    int sim_length = 100;
    int takeover_timestep = 1;
    int ego_id = 0;
    int fov = 200;
    int width = 200;
    int height = 200;
    std::optional<float> scenario_center_x = std::nullopt;
    std::optional<float> scenario_center_y = std::nullopt;
    bool get_infractions = false;
    std::string model_version_drive = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--log-path" && i + 1 < argc) {
            log_path = argv[++i];
        } else if (arg == "--num-agents" && i + 1 < argc) {
            num_agents = std::stoi(argv[++i]);
        } else if (arg == "--sim-length" && i + 1 < argc) {
            sim_length = std::stoi(argv[++i]);
        } else if (arg == "--takeover-timestep" && i + 1 < argc) {
            takeover_timestep = std::stoi(argv[++i]);
        } else if (arg == "--ego-id" && i + 1 < argc) {
            ego_id = std::stoi(argv[++i]);
        } else if (arg == "--fov" && i + 1 < argc) {
            fov = std::stoi(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            height = std::stoi(argv[++i]);
        } else if (arg == "--scenario-center-x" && i + 1 < argc) {
            scenario_center_x = std::stof(argv[++i]);
        } else if (arg == "--scenario-center-y" && i + 1 < argc) {
            scenario_center_y = std::stof(argv[++i]);
        } else if (arg == "--get-infractions") {
            get_infractions = true;
        } else if (arg == "--model-version-drive" && i + 1 < argc) {
            model_version_drive = argv[++i];
        } else {
            std::cerr << "Unknown or malformed argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    const char *api_key_env = std::getenv("IAI_API_KEY");
    if (api_key_env == nullptr || std::string(api_key_env).empty()) {
        std::cerr << "IAI_API_KEY environment variable is not set.\n";
        return 1;
    }
    const std::string API_KEY = api_key_env;

    std::optional<std::string> model_version_or_nullopt = std::nullopt;
    if (!model_version_drive.empty() && model_version_drive != "None") {
        model_version_or_nullopt = model_version_drive;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    int seed = std::uniform_int_distribution<>(
        1,
        1000000
    )(gen);

    boost::asio::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    Session session(
        ioc,
        ctx
    );
    session.set_api_key(API_KEY);
    session.connect();

    std::cout << "Reading scenario log from: " << log_path << "\n";
    ScenarioLogReader log_reader(log_path);
    const std::string location = log_reader.get_location();

    log_reader.initialize();

    const std::vector<AgentProperties> log_agent_properties = log_reader.current_agent_properties();
    const std::vector<AgentState> log_agent_states_t0 = log_reader.current_agent_states();
    const std::optional<std::map<std::string, std::string>> log_traffic_lights_t0 = log_reader.current_traffic_lights();

    const size_t NUM_LOG_AGENTS = log_agent_properties.size();
    const int LOG_LENGTH = log_reader.get_scenario_length();
    std::cout << "Log location: " << location << "; log agents: " << NUM_LOG_AGENTS << "; log length: " << LOG_LENGTH << "\n";

    if (ego_id < 0 || static_cast<size_t>(ego_id) >= NUM_LOG_AGENTS) {
        std::cerr << "--ego-id " << ego_id << " is out of range; log has " << NUM_LOG_AGENTS << " agents.\n";
        return 1;
    }

    LocationInfoRequest li_req("{}");
    li_req.set_location(location);
    li_req.set_include_map_source(true);
    li_req.set_rendering_fov(fov);
    if (scenario_center_x.has_value() && scenario_center_y.has_value()) {
        li_req.set_rendering_center(std::make_pair<double, double>(
            *scenario_center_x,
            *scenario_center_y
        ));
    }
    LocationInfoResponse li_res = location_info(
        li_req,
        &session
    );

    std::pair<float, float> scenario_center;
    if (scenario_center_x.has_value() && scenario_center_y.has_value()) {
        scenario_center = std::pair<float, float>{*scenario_center_x, *scenario_center_y};
    } else {
        scenario_center = std::pair<float, float>{
            static_cast<float>(li_res.map_origin().x),
            static_cast<float>(li_res.map_origin().y)};
    }
    std::cout << "Scenario center: (" << scenario_center.first << ", " << scenario_center.second << ")\n";

    std::vector<Region> regions;
    if (num_agents > 0) {
        std::map<AgentType, int> agent_count_dict = {{AgentType::car, num_agents}};
        regions = get_regions_default(
            location,
            num_agents,
            agent_count_dict,
            session,
            std::pair<float, float>{width / 2.f, height / 2.f},
            scenario_center,
            seed
        );
        std::cout << "Generated " << regions.size() << " regions for " << num_agents << " sampled NPCs.\n";
    } else {
        Point2d center{scenario_center.first, scenario_center.second};
        regions.push_back(Region::create_square_region(
            center,
            static_cast<double>(fov)
        ));
        std::cout << "No NPCs requested; using a single square region of size " << fov << " around the scenario center.\n";
    }

    LargeInitializeConfig init_cfg(session);
    init_cfg.location = location;
    init_cfg.regions = regions;
    init_cfg.agent_properties = log_agent_properties;
    init_cfg.agent_states = log_agent_states_t0;
    init_cfg.traffic_light_state_history = log_traffic_lights_t0;
    init_cfg.get_infractions = get_infractions;
    init_cfg.random_seed = seed;
    init_cfg.api_model_version = model_version_or_nullopt;
    init_cfg.return_exact_agents = true;

    LargeInitializeConfig reinit_cfg(session);
    reinit_cfg.location = location;
    reinit_cfg.regions = {Region::create_square_region(
        Point2d{scenario_center.first, scenario_center.second},
        std::max({
            static_cast<double>(width),
            static_cast<double>(height),
            static_cast<double>(fov)
        })
    )};
    reinit_cfg.get_infractions = get_infractions;
    reinit_cfg.random_seed = seed;
    reinit_cfg.api_model_version = model_version_or_nullopt;
    reinit_cfg.return_exact_agents = true;

    std::cout << "Calling large_initialize...\n";
    InitializeResponse init_response = large_initialize(init_cfg);

    // NOTE: the Python example refreshes waypoints via iai.WaypointManager here;
    // the C++ SDK does not currently expose a WaypointManager, so this step is omitted.

    std::vector<AgentState> agent_states = init_response.agent_states();
    std::vector<AgentProperties> agent_properties = init_response.agent_properties();
    std::vector<std::vector<double>> recurrent_states = init_response.recurrent_states();
    std::optional<std::vector<LightRecurrentState>> light_recurrent_states = init_response.light_recurrent_states();

    const size_t total_num_agents = agent_states.size();
    std::cout << "Number of agents in simulation: " << total_num_agents << "\n";

    ScenarioLogWriter log_writer;
    log_writer.initialize(
        location,
        li_res,
        init_response,
        std::nullopt,
        seed,
        seed,
        model_version_or_nullopt,
        std::nullopt
    );

    bool flip_x_for_carla = (location.rfind("carla:", 0) == 0);
    ScenePlotter scene_plotter(
        li_res,
        flip_x_for_carla,
        static_cast<double>(fov),
        2048
    );
    const std::string video_path = "scenario_replay_ego_takeover.mp4";
    scene_plotter.initialize_video(
        video_path,
        10
    );

    std::cout << "Stepping through simulation for " << sim_length << " timesteps...\n";
    for (int ts = 0; ts < sim_length; ++ts) {
        std::optional<std::map<std::string, std::string>> current_log_lights = log_reader.current_traffic_lights();
        bool is_log_lights = current_log_lights.has_value() && ts < LOG_LENGTH;

        LargeDriveConfig drive_cfg(session);
        drive_cfg.location = location;
        drive_cfg.api_key = API_KEY;
        drive_cfg.agent_states = agent_states;
        drive_cfg.agent_properties = agent_properties;
        drive_cfg.recurrent_states = recurrent_states;
        drive_cfg.traffic_lights_states = is_log_lights ? current_log_lights : std::nullopt;
        drive_cfg.light_recurrent_states = is_log_lights ? std::nullopt : light_recurrent_states;
        drive_cfg.random_seed = seed;
        drive_cfg.get_infractions = get_infractions;
        drive_cfg.api_model_version = model_version_or_nullopt;
        drive_cfg.single_call_agent_limit = 100;
        drive_cfg.async_api_calls = true;

        DriveResponse drive_response = large_drive(drive_cfg);

        agent_states = drive_response.agent_states();
        recurrent_states = drive_response.recurrent_states();
        light_recurrent_states = drive_response.light_recurrent_states();
        std::optional<std::map<std::string, std::string>> traffic_lights_out = drive_response.traffic_lights_states();

        // Force log-agent positions back onto the log trajectory; once the ego
        // takeover timestep is reached, leave the ego at the API-produced state
        // so it can react to the surrounding NPCs.
        if (ts < LOG_LENGTH) {
            log_reader.drive();
            const std::vector<AgentState> log_states = log_reader.current_agent_states();
            const size_t bound = std::min(
                NUM_LOG_AGENTS,
                log_states.size()
            );
            for (size_t i = 0; i < bound; ++i) {
                if (static_cast<int>(i) == ego_id && ts >= takeover_timestep) {
                    continue;
                }
                agent_states[i] = log_states[i];
            }
            drive_response.set_agent_states(agent_states);
        }

        log_writer.drive(drive_response);
        scene_plotter.render_step(
            agent_states,
            agent_properties,
            traffic_lights_out
        );

        std::vector<std::size_t> agent_ids_to_remove;
        std::vector<AgentState> agent_states_to_add;
        std::vector<AgentProperties> agent_properties_to_add;
        update_agent_population(
            agent_states,
            agent_properties,
            recurrent_states,
            reinit_cfg,
            agent_ids_to_remove,
            agent_states_to_add,
            agent_properties_to_add
        );

        if ((ts + 1) % 10 == 0 || ts + 1 == sim_length) {
            std::cout << "  step " << (ts + 1) << "/" << sim_length << "\n";
        }
    }

    scene_plotter.close();

    const std::string output_log_path = "scenario_replay_ego_takeover_output.json";
    log_writer.export_to_file(
        output_log_path,
        std::nullopt,
        li_res
    );

    std::cout << "\nDone. Created:\n"
              << "  - " << video_path << "\n"
              << "  - " << output_log_path << "\n";
    return 0;
}
