#include "large_initialize.h"
#include "externals/json.hpp"
#include "initialize_response.h" // Ensure this header defines InitializeResponse
#include <random>


namespace invertedai {

using RegionMap = std::vector<std::pair<int, int>>;

double squared_distance(const Point2d& p1, const Point2d& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

std::pair<std::vector<Region>, RegionMap> insert_agents_into_nearest_regions(
    std::vector<Region> regions,
    const std::vector<AgentProperties>& agent_properties,
    const std::vector<AgentState>& agent_states,
    bool return_region_index = false,
    std::optional<int> random_seed = std::nullopt
) {
    size_t num_agent_states = agent_states.size();
    size_t num_regions = regions.size();
    size_t num_agent_properties = agent_properties.size();
    if(num_regions == 0) {
        throw std::invalid_argument("Invalid parameter: number of regions must be greater than zero.");
    }
    if(num_agent_properties >= num_agent_states) {
        throw std:: invalid_argument("Invalid parameters: number of agent properties must be greater than number of agent states.");
    }

    RegionMap region_map;

    // track how many states have been assigned to each region
    std::vector<size_t> region_agent_state_lengths;
    region_agent_state_lengths.reserve(num_regions); // for performance
    for(const auto& r : regions ) {
        region_agent_state_lengths.push_back(r.agent_states.size());
    }
    // place predefined agents with states into nearest regions
    for(size_t i = 0; i < num_agent_states; i++) {
        const auto& prop = agent_properties[i];
        const auto& state = agent_states[i];

        // find nearest region                                   //// very inefficient???
        double min_dist = std::numeric_limits<double>::max();
        int closest_idx = -1;
        Point2d agent_center{state.x, state.y};
        for(size_t r = 0; r< num_regions; r++) {
            const auto& region = regions[r];
            double dist = squared_distance(agent_center, region.center);
            if(dist < min_dist) {
                min_dist = dist;
                closest_idx = (int) r;
            }
        }
        // insert agent into nearest region
        if(closest_idx == -1) {
            throw std::runtime_error("Failed to find closest region for agent.");
        }

        int insert_index = static_cast<int>(region_agent_state_lengths[closest_idx]);
        region_agent_state_lengths[closest_idx]++;



        regions[closest_idx].agent_states.insert(
            regions[closest_idx].agent_states.begin() + insert_index, state);
        regions[closest_idx].agent_properties.insert(
            regions[closest_idx].agent_properties.begin() + insert_index, prop);

        if (return_region_index) {
            region_map.emplace_back(closest_idx, insert_index);
        }
    }
    
        // 2. Place remaining properties (agents without states)
        std::mt19937 rng(random_seed.value_or(std::random_device{}()));
        std::uniform_int_distribution<int> dist(0, static_cast<int>(num_regions - 1));
    
        for (size_t i = num_agent_states; i < agent_properties.size(); i++) {
            int rand_idx = dist(rng);
            regions[rand_idx].agent_properties.push_back(agent_properties[i]);
    
            if (return_region_index) {
                region_map.emplace_back(rand_idx, static_cast<int>(regions[rand_idx].agent_properties.size() - 1));
            }
        }
    
        return {regions, region_map};
    }

    static void get_all_existing_agents_from_regions(
        const std::vector<Region>& regions,
        std::size_t exclude_index,
        const Region& nearby_region,
        std::vector<AgentState>& out_states,
        std::vector<AgentProperties>& out_props) {
    
      out_states.clear();
      out_props.clear();
    
      for (std::size_t i = 0; i < regions.size(); ++i) {
        if (i == exclude_index) continue;
    
        const Region& r = regions[i];
        if (std::sqrt(squared_distance(nearby_region.center, r.center)) >
            (REGION_MAX_SIZE + AGENT_SCOPE_FOV_BUFFER)) {
          continue;
        }
    
        const auto n = std::min(r.agent_states.size(), r.agent_properties.size());
        out_states.insert(out_states.end(), r.agent_states.begin(), r.agent_states.begin() + n);
        out_props.insert(out_props.end(), r.agent_properties.begin(), r.agent_properties.begin() + n);
      }
    }
    
    inline bool inside_fov(const Point2d& center, double fov, const Point2d& p) {
    return (center.x - fov / 2.0 <= p.x && p.x <= center.x + fov / 2.0 &&
            center.y - fov / 2.0 <= p.y && p.y <= center.y + fov / 2.0);
    }
    std::pair<std::vector<Region>, std::vector<InitializeResponse>>
    initialize_regions(
        const std::string& location,
        std::vector<Region> regions,
        const std::optional<std::vector<TrafficLightState>>& traffic_light_state_history,
        bool get_infractions,
        const std::optional<int>& random_seed,
        const std::optional<std::string>& api_model_version,
        bool /*display_progress_bar*/,
        bool return_exact_agents) {
    
      std::vector<InitializeResponse> all_responses;
      const int num_attempts = 1 + static_cast<int>(regions.size()) / ATTEMPT_PER_NUM_REGIONS;
    
      for (std::size_t i = 0; i < regions.size(); ++i) {
        Region& region = regions[i];
    
        // collect conditional agents from neighbours near FOV
        std::vector<AgentState> neighbour_states, neighbour_props_states;
        std::vector<AgentState> existing_states;
        std::vector<AgentProperties> existing_props;
        get_all_existing_agents_from_regions(regions, i, region, existing_states, existing_props);
    
        std::vector<AgentState> out_states;
        std::vector<AgentProperties> out_props;
        out_states.reserve(existing_states.size());
        out_props.reserve(existing_props.size());
    
        for (std::size_t k = 0; k < existing_states.size(); ++k) {
          const auto& s = existing_states[k];
          if (inside_fov(region.center, region.size + AGENT_SCOPE_FOV_BUFFER, {s.x, s.y})) {
            out_states.push_back(s);
            out_props.push_back(existing_props[k]);
          }
        }
    
        // Combine out-of-region conditionals + region’s own agents
        std::vector<AgentState> all_states = out_states;
        std::vector<AgentProperties> all_props = out_props;
    
        all_states.insert(all_states.end(), region.agent_states.begin(), region.agent_states.end());
        all_props.insert(all_props.end(), region.agent_properties.begin(), region.agent_properties.end());
    
        const std::size_t num_out_cond = out_states.size();
        const std::size_t num_region_cond = region.agent_states.size();
    
        // clear the region for re-fill after API returns
        region.clear_agents();
    
        if (all_props.empty()) continue;
    
        // Try initialize with retries
        InitializeResponse resp;
        bool success = false;
        for (int attempt = 0; attempt < num_attempts; ++attempt) {
          try {
            // resp = initialize_api_call(
            //     location,
            //     all_states,            // one-timestep history (flattened)
            //     all_props,
            //     get_infractions,
            //     traffic_light_state_history,
            //     region.center,
            //     random_seed,
            //     api_model_version);
            // success = true;
            break;
          } catch (...) {
            // log & retry
          }
        }
    
        if (!success) {
          if (return_exact_agents) {
            throw std::runtime_error("Unable to initialize region " + std::to_string(i));
          } else {
            // partial fallback: only recurrent for predefined agents (if you want)
            continue;
          }
        }
    
        // 4) Filter out conditionals from other regions and keep only in-FOV (unless strict)
        InitializeResponse filtered;
        for (std::size_t j = num_out_cond; j < resp.agent_states.size(); ++j) {
          const auto& s = resp.agent_states[j];
          const auto& p = resp.agent_properties[j];
          const auto& r = resp.recurrent_states[j];
    
          if (!return_exact_agents &&
              !inside_fov(region.center, region.size, {s.x, s.y})) {
            continue;
          }
    
          region.insert_agent(s, p, r);
    
          filtered.agent_states.push_back(s);
          filtered.agent_properties.push_back(p);
          filtered.recurrent_states.push_back(r);
          if (get_infractions && j < resp.infractions.size())
            filtered.infractions.push_back(resp.infractions[j]);
        }
    
        all_responses.push_back(std::move(filtered));
      }
    
      return {std::move(regions), std::move(all_responses)};
    }


InitializeResponse large_initialize(const LargeInitializeConfig& cfg) {
    // validate inputs
    if(cfg.regions.size() == 0) {
        throw std::invalid_argument("At least one region must be provided.");
    }
    if (cfg.agent_states.has_value()) {
        if (!cfg.agent_properties.has_value() ||
            cfg.agent_properties->size() < cfg.agent_states->size()) {
            throw std::invalid_argument(
                "Invalid parameters: number of agent properties must be greater than number of agent states."
            );
        }
    }
    // Insert agents into nearest regions (preserve indices)
    auto [regions_with_agents, region_map] = invertedai::insert_agents_into_nearest_regions(
        cfg.regions,
        cfg.agent_properties.value_or(std::vector<AgentProperties>{}),
        cfg.agent_states.value_or(std::vector<AgentState>{}),
        true,  // return_region_index
        cfg.random_seed
    );

    // Call Initialize for each region
    auto [final_regions, all_reponses] = invertedai::initialize_regions(
        cfg.location,
        regions_with_agents,
        cfg.traffic_light_state_history,
        cfg.get_infractions,
        cfg.random_seed,
        cfg.api_model_version,
        cfg.display_progress_bar,
        cfg.return_exact_agents
    );

    // consolidate responses
    InitializeResponse response = invertedai::consolidate_all_responses(
        all_reponses,
        region_map,
        cfg.return_exact_agents,
        cfg.get_infractions
    );

    return response;
}
} // namespace invertedai