#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "invertedai/api.h"
#include "invertedai/data_utils.h"
#include "invertedai/initialize_response.h"
#include "invertedai/location_info_request.h"
#include "invertedai/location_info_response.h"
#include "invertedai/session.h"
#include "invertedai/visualize.h"
#include "large/large_drive/large_drive.h"
#include "large/large_initialize/large_init_helpers.h"
#include "large/large_initialize/large_initialize.h"

using namespace invertedai;

/*
    HOW TO RUN EXECUTABLE:

    1. cd into invertedai_cpp/
    2. docker compose run --rm dev
    3. export IAI_API_KEY="your_key_here"
    4. bazel build //examples:changing_agents_example
    5. ./bazel-bin/examples/changing_agents_example --help
       ./bazel-bin/examples/changing_agents_example \
           --location carla:Town03 --num-agents 10 --timesteps-per-stage 20

    WHAT THIS EXAMPLE SHOWS:

    A client integration where the agent population CHANGES across the
    simulation. The example walks through four stages:

      1. Spawn N initial agents (CLIENT-OWNED step — see spawn_agents_with_iai).
      2. Acquire recurrent_states for those agents and drive for K steps.
      3. Add 1 agent (same client-owned spawn step) and drive for K steps.
      4. Remove 1 agent at random and drive for K steps.

    Stages 2, 3 and 4 each call acquire_recurrent_states() after the population
    changes. That call is the load-bearing piece you must replicate in your
    own integration. The spawn helper is a stand-in — replace it with your
    own world generator.

    A single random seed is shared across all API calls and the random-removal
    pick so the run is reproducible from one launch to the next.
*/

struct Cli {
  std::string location = "carla:Town03";
  int num_agents = 10;
  int timesteps_per_stage = 20;
  std::optional<double> region_center_x = std::nullopt;
  std::optional<double> region_center_y = std::nullopt;
  double region_size = 100.0;
  bool get_infractions = false;
  std::string output = "changing_agents_demo.avi";
};

static void print_usage(
    const char* bin
) {
  std::cout
      << "Usage: " << bin << " [options]\n\n"
      << "Demonstrates a client integration where the agent population\n"
      << "changes between time steps over four stages: spawn, drive, add,\n"
      << "drive, remove, drive.\n\n"
      << "Options:\n"
      << "  --location <str>             Map location (default: carla:Town03)\n"
      << "  --num-agents <int>           Initial agent count (default: 10)\n"
      << "  --timesteps-per-stage <int>  Drive steps per stage (default: 20)\n"
      << "  --region-center-x <float>    Region center x (default: location_info().map_origin().x)\n"
      << "  --region-center-y <float>    Region center y (default: location_info().map_origin().y)\n"
      << "  --region-size <float>        Square region edge length in meters (default: 100)\n"
      << "  --get-infractions            Capture infraction data (default: off)\n"
      << "  --output <str>               Output video filename (default: changing_agents_demo.avi)\n"
      << "  --help, -h                   Print this message\n";
}

static Cli parse_args(
    int argc,
    char** argv
) {
  Cli cli;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg == "--location" && i + 1 < argc) {
      cli.location = argv[++i];
    } else if (arg == "--num-agents" && i + 1 < argc) {
      cli.num_agents = std::stoi(argv[++i]);
    } else if (arg == "--timesteps-per-stage" && i + 1 < argc) {
      cli.timesteps_per_stage = std::stoi(argv[++i]);
    } else if (arg == "--region-center-x" && i + 1 < argc) {
      cli.region_center_x = std::stod(argv[++i]);
    } else if (arg == "--region-center-y" && i + 1 < argc) {
      cli.region_center_y = std::stod(argv[++i]);
    } else if (arg == "--region-size" && i + 1 < argc) {
      cli.region_size = std::stod(argv[++i]);
    } else if (arg == "--get-infractions") {
      cli.get_infractions = true;
    } else if (arg == "--output" && i + 1 < argc) {
      cli.output = argv[++i];
    } else {
      std::cerr << "Unknown or malformed argument: " << arg << "\n";
      print_usage(argv[0]);
      std::exit(1);
    }
  }
  return cli;
}

// =============================================================================
// HELPER A — STAND-IN for the integrator's own world generator.
// =============================================================================
//
// In a real integration, your simulator decides where new agents appear and
// what their initial state (x, y, orientation, speed) is. This example uses
// large_initialize() purely as a convenient way to produce plausible agent
// states for the demo. REPLACE THE BODY OF THIS FUNCTION with calls into
// your own world.
//
// `existing_states` and `existing_properties` are the agents already present
// in the world (pass empty vectors when bootstrapping). `num_agents_to_add`
// is the count of NEW agents to introduce. Returns (states, properties) for
// the COMBINED population (existing + new) with new agents appended at the
// end.
//
// Recurrent state is intentionally NOT returned here — a real client-owned
// simulator wouldn't have one. Acquire it via acquire_recurrent_states()
// (helper B) once the population is finalized.
static std::pair<std::vector<AgentState>, std::vector<AgentProperties>>
spawn_agents_with_iai(
    Session& session,
    const std::string& location,
    const Point2d& region_center,
    double region_size,
    int num_agents_to_add,
    const std::vector<AgentState>& existing_states,
    const std::vector<AgentProperties>& existing_properties,
    int random_seed
) {
  std::vector<AgentProperties> properties = existing_properties;
  for (int i = 0; i < num_agents_to_add; ++i) {
    properties.push_back(make_default_properties(AgentType::car));
  }

  // Single square region. Properties without matching states are spawned by
  // the API (see LargeInitializeConfig::agent_properties docstring).
  Region region = Region::create_square_region(
      region_center,
      region_size
  );

  LargeInitializeConfig cfg(session);
  cfg.location = location;
  cfg.regions = {region};
  cfg.random_seed = random_seed;
  cfg.return_exact_agents = false;
  cfg.traffic_light_state_history = std::nullopt;
  if (existing_states.empty()) {
    cfg.agent_states = std::nullopt;
  } else {
    cfg.agent_states = existing_states;
  }
  cfg.agent_properties = properties;

  InitializeResponse response = large_initialize(cfg);
  return {
      response.agent_states(),
      response.agent_properties()
  };
}

// =============================================================================
// HELPER B — load-bearing: refresh recurrent_states for the current population.
// =============================================================================
//
// This is the workflow you DO want to copy. Whenever the agent population
// changes (after a spawn or a removal), call large_initialize() with
// return_exact_agents=true and the FULL current agent_states +
// agent_properties so the API recomputes a recurrent_states vector that is
// consistent across the new population. Skipping this step after a
// population change leads to undefined drive() behavior for agents whose
// recurrent state is stale.
//
// Returns the full InitializeResponse so the caller can read recurrent_states,
// traffic_lights_states, and light_recurrent_states off it.
static InitializeResponse acquire_recurrent_states(
    Session& session,
    const std::string& location,
    const Point2d& region_center,
    double region_size,
    const std::vector<AgentState>& agent_states,
    const std::vector<AgentProperties>& agent_properties,
    int random_seed
) {
  Region region = Region::create_square_region(
      region_center,
      region_size
  );

  LargeInitializeConfig cfg(session);
  cfg.location = location;
  cfg.regions = {region};
  cfg.random_seed = random_seed;
  cfg.return_exact_agents = true;
  cfg.traffic_light_state_history = std::nullopt;
  cfg.agent_states = agent_states;
  cfg.agent_properties = agent_properties;

  return large_initialize(cfg);
}

// =============================================================================
// HELPER C — single large_drive() step plus the bookkeeping it requires.
// =============================================================================
//
// Calls large_drive() once and feeds agent_states, recurrent_states, and
// light_recurrent_states back into the LargeDriveConfig for the next
// iteration. The drive_cfg is mutated in place; the response is returned so
// the caller can render its traffic_lights_states.
static DriveResponse step_simulation(
    LargeDriveConfig& drive_cfg
) {
  DriveResponse response = large_drive(drive_cfg);
  drive_cfg.agent_states = response.agent_states();
  drive_cfg.recurrent_states = response.recurrent_states();
  drive_cfg.light_recurrent_states = response.light_recurrent_states();
  drive_cfg.traffic_lights_states = std::nullopt;
  drive_cfg.api_model_version = response.model_version();
  drive_cfg.random_seed = std::nullopt;
  return response;
}

// =============================================================================
// HELPER D — packaging boilerplate so main() reads top-to-bottom as 4 stages.
// =============================================================================

static LargeDriveConfig build_drive_config(
    Session& session,
    const std::string& api_key,
    const std::string& location,
    int random_seed,
    bool get_infractions
) {
  LargeDriveConfig cfg(session);
  cfg.location = location;
  cfg.api_key = api_key;
  cfg.random_seed = random_seed;
  cfg.get_infractions = get_infractions;
  return cfg;
}

// Push the (possibly new) agent list and recurrent_states into drive_cfg, plus
// the traffic light fields from the most recent InitializeResponse so DRIVE
// has a consistent starting point.
static void seed_drive_config_from_initialize(
    LargeDriveConfig& drive_cfg,
    const std::vector<AgentState>& agent_states,
    const std::vector<AgentProperties>& agent_properties,
    const std::vector<std::vector<double>>& recurrent_states,
    const InitializeResponse& init_resp
) {
  drive_cfg.agent_states = agent_states;
  drive_cfg.agent_properties = agent_properties;
  drive_cfg.recurrent_states = recurrent_states;
  drive_cfg.traffic_lights_states = init_resp.traffic_lights_states();
  drive_cfg.light_recurrent_states = init_resp.light_recurrent_states();
}

static void run_drive_loop(
    LargeDriveConfig& drive_cfg,
    ScenePlotter& plotter,
    int n_steps,
    const std::string& stage_label
) {
  std::cout
      << "[" << stage_label << "] driving "
      << drive_cfg.agent_states.size() << " agents for "
      << n_steps << " steps...\n";
  for (int t = 0; t < n_steps; ++t) {
    DriveResponse response = step_simulation(drive_cfg);
    plotter.render_step(
        drive_cfg.agent_states,
        drive_cfg.agent_properties,
        response.traffic_lights_states()
    );
    std::cout << "  step " << (t + 1) << "/" << n_steps << "\n";
  }
}

// =============================================================================
// main — four labeled stages.
// =============================================================================

int main(
    int argc,
    char** argv
) {
  Cli cli = parse_args(
      argc,
      argv
  );

  const char* api_key_env = std::getenv("IAI_API_KEY");
  if (api_key_env == nullptr || std::string(api_key_env).empty()) {
    std::cerr << "ERROR: IAI_API_KEY environment variable is not set.\n";
    return 1;
  }
  const std::string api_key = api_key_env;

  // Single seed shared across every API call and the random-removal pick.
  std::random_device rd;
  std::mt19937 gen(rd());
  const int seed = std::uniform_int_distribution<>(
      1,
      1000000
  )(gen);

  boost::asio::io_context ioc;
  ssl::context ctx(ssl::context::tlsv12_client);
  Session session(
      ioc,
      ctx
  );
  session.set_api_key(api_key);
  session.connect();

  // Resolve region center: CLI overrides; otherwise fall back to map_origin.
  // The probe call is skipped when both coords come from CLI.
  double region_cx;
  double region_cy;
  if (cli.region_center_x.has_value() && cli.region_center_y.has_value()) {
    region_cx = *cli.region_center_x;
    region_cy = *cli.region_center_y;
  } else {
    LocationInfoRequest probe_req("{}");
    probe_req.set_location(cli.location);
    LocationInfoResponse probe = location_info(
        probe_req,
        &session
    );
    region_cx = cli.region_center_x.value_or(probe.map_origin().x);
    region_cy = cli.region_center_y.value_or(probe.map_origin().y);
  }
  Point2d region_center{region_cx, region_cy};
  std::cout
      << "Location: " << cli.location
      << " | region center: (" << region_cx << ", " << region_cy
      << ") | region size: " << cli.region_size << " m"
      << " | seed: " << seed << "\n";

  // Second location_info call — this one is framed on the region so the
  // ScenePlotter background lines up with the agents we will render.
  LocationInfoRequest li_req("{}");
  li_req.set_location(cli.location);
  li_req.set_rendering_center(
      std::make_pair(region_cx, region_cy)
  );
  li_req.set_rendering_fov(static_cast<int>(std::lround(cli.region_size)));
  li_req.set_include_map_source(true);
  LocationInfoResponse li_res = location_info(
      li_req,
      &session
  );

  bool flip_x = (cli.location.rfind("carla:", 0) == 0);
  ScenePlotter plotter(
      li_res,
      flip_x,
      cli.region_size
  );
  plotter.initialize_video(
      cli.output,
      10
  );

  // -----------------------------------------------------------------------
  // STAGE 1: bootstrap population.
  //
  // CLIENT-OWNED step. spawn_agents_with_iai() is a stand-in for the
  // integrator's own world generator — replace its body in a real
  // integration. We treat its outputs as if they came from the integrator's
  // simulator: only agent_states and agent_properties cross the boundary,
  // recurrent_states are intentionally discarded.
  // -----------------------------------------------------------------------
  std::cout
      << "\n=== Stage 1: spawn " << cli.num_agents
      << " initial agents (client stand-in) ===\n";
  auto [agent_states, agent_properties] = spawn_agents_with_iai(
      session,
      cli.location,
      region_center,
      cli.region_size,
      cli.num_agents,
      /*existing_states=*/{},
      /*existing_properties=*/{},
      seed
  );

  // -----------------------------------------------------------------------
  // STAGE 2: acquire recurrent_states for the bootstrap population, drive.
  //
  // Required after every population change. acquire_recurrent_states()
  // calls large_initialize() with return_exact_agents=true so the API does
  // not alter the agent list — it just hands back consistent
  // recurrent_states for the agents we passed in.
  // -----------------------------------------------------------------------
  std::cout << "\n=== Stage 2: acquire recurrent_states + drive ===\n";
  InitializeResponse init_resp = acquire_recurrent_states(
      session,
      cli.location,
      region_center,
      cli.region_size,
      agent_states,
      agent_properties,
      seed
  );

  LargeDriveConfig drive_cfg = build_drive_config(
      session,
      api_key,
      cli.location,
      seed,
      cli.get_infractions
  );
  seed_drive_config_from_initialize(
      drive_cfg,
      agent_states,
      agent_properties,
      init_resp.recurrent_states(),
      init_resp
  );

  run_drive_loop(
      drive_cfg,
      plotter,
      cli.timesteps_per_stage,
      "Stage 2"
  );

  agent_states = drive_cfg.agent_states;
  agent_properties = drive_cfg.agent_properties;

  // -----------------------------------------------------------------------
  // STAGE 3: add 1 agent, refresh recurrent_states, drive.
  //
  // The spawn step here is the same convenience-shortcut as Stage 1 —
  // replace it with your own logic in a real integration. The
  // acquire_recurrent_states() call that follows IS the load-bearing step:
  // it produces a recurrent_states vector that is consistent across the
  // whole population (existing agents + the new one).
  // -----------------------------------------------------------------------
  std::cout
      << "\n=== Stage 3: add 1 agent (now "
      << (agent_states.size() + 1) << " total) ===\n";
  std::tie(agent_states, agent_properties) = spawn_agents_with_iai(
      session,
      cli.location,
      region_center,
      cli.region_size,
      /*num_agents_to_add=*/1,
      agent_states,
      agent_properties,
      seed
  );

  init_resp = acquire_recurrent_states(
      session,
      cli.location,
      region_center,
      cli.region_size,
      agent_states,
      agent_properties,
      seed
  );

  seed_drive_config_from_initialize(
      drive_cfg,
      agent_states,
      agent_properties,
      init_resp.recurrent_states(),
      init_resp
  );

  run_drive_loop(
      drive_cfg,
      plotter,
      cli.timesteps_per_stage,
      "Stage 3"
  );

  agent_states = drive_cfg.agent_states;
  agent_properties = drive_cfg.agent_properties;

  // -----------------------------------------------------------------------
  // STAGE 4: remove 1 agent at random, refresh recurrent_states, drive.
  //
  // We erase the chosen index from all three local lists FIRST so they stay
  // aligned, then call acquire_recurrent_states() to recompute
  // recurrent_states for the now-smaller population. The popped entry of
  // recurrent_states is discarded — it belonged to the removed agent.
  // -----------------------------------------------------------------------
  if (agent_states.empty()) {
    std::cerr << "ERROR: no agents left to remove.\n";
    plotter.close();
    return 1;
  }
  std::vector<std::vector<double>> recurrent_states = drive_cfg.recurrent_states.value();
  std::size_t remove_idx = std::uniform_int_distribution<std::size_t>(
      0,
      agent_states.size() - 1
  )(gen);
  std::cout
      << "\n=== Stage 4: remove agent at index " << remove_idx
      << " (now " << (agent_states.size() - 1) << " total) ===\n";
  agent_states.erase(agent_states.begin() + remove_idx);
  agent_properties.erase(agent_properties.begin() + remove_idx);
  recurrent_states.erase(recurrent_states.begin() + remove_idx);

  init_resp = acquire_recurrent_states(
      session,
      cli.location,
      region_center,
      cli.region_size,
      agent_states,
      agent_properties,
      seed
  );

  seed_drive_config_from_initialize(
      drive_cfg,
      agent_states,
      agent_properties,
      init_resp.recurrent_states(),
      init_resp
  );

  run_drive_loop(
      drive_cfg,
      plotter,
      cli.timesteps_per_stage,
      "Stage 4"
  );

  plotter.close();
  std::cout << "\nWrote " << cli.output << "\n";
  return 0;
}
