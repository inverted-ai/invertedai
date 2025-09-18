#pragma once
#include "large_initialize.h"
#include "invertedai/initialize_response.h" // Ensure this header defines InitializeResponse
#include <random>
#include "invertedai/initialize_request.h"
#include "../invertedai/api.h"
#include "invertedai/error.h"
using tcp = net::ip::tcp;    // from <boost/asio/ip/tcp.hpp>
using json = nlohmann::json; // from <json.hpp>

namespace invertedai {

using RegionMap = std::vector<std::pair<int, int>>;

double squared_distance(const Point2d& p1, const Point2d& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

std::vector<std::map<std::string,std::string>>
convert_traffic_light_history(const std::vector<TrafficLightStatesDict>& dicts) {
    std::vector<std::map<std::string,std::string>> result;
    for (const auto& states : dicts) {
        std::map<std::string,std::string> entry;
        for (const auto& st : states) {
            entry[st.id] = st.value; // assuming TrafficLightState has .id and .state
        }
        result.push_back(entry);
    }
    return result;
}

std::pair<std::vector<Region>, RegionMap>   insert_agents_into_nearest_regions(
    std::vector<invertedai::Region> regions,
    const std::vector<AgentProperties>& agent_properties,
    const std::vector<AgentState>& agent_states,
    bool return_region_index,
    std::optional<int> random_seed
) {
    size_t num_agent_states = agent_states.size();
    size_t num_regions = regions.size();
    size_t num_agent_properties = agent_properties.size();
    if(num_regions <= 0) {
        throw std::invalid_argument("Invalid parameter: number of regions must be greater than zero.");
    }
    if(num_agent_properties < num_agent_states) {
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
        if (i >= num_agent_properties) {
            throw std::invalid_argument("Not enough agent properties for the given agent states.");
        }

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

    static void pick_borrowed_agents(
        const std::vector<AgentState>&  filtered_states,
        const std::vector<AgentProperties>& filtered_props,
        size_t max_take,
        std::optional<int> random_seed,
        std::vector<AgentState>& out_states,
        std::vector<AgentProperties>& out_props
    ) {
        out_states.clear();
        out_props.clear();
        const size_t n = std::min(filtered_states.size(), filtered_props.size());
        if (n == 0) return;
    
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
    
        std::mt19937 rng(random_seed.value_or(std::random_device{}()));
        std::shuffle(idx.begin(), idx.end(), rng);
    
        const size_t take = std::min(max_take, n);
        out_states.reserve(take);
        out_props.reserve(take);
        for (size_t k = 0; k < take; ++k) {
            size_t j = idx[k];
            out_states.push_back(filtered_states[j]);
            out_props.push_back(filtered_props[j]);
        }
    }
    
    std::pair<std::vector<invertedai::Region>, std::vector<invertedai::InitializeResponse>>
    initialize_regions(
        const std::string& location,
        std::vector<invertedai::Region> regions,
        const std::optional<std::vector<TrafficLightStatesDict>>& traffic_light_state_history,
        bool get_infractions,
        std::optional<int> random_seed,
        std::optional<std::string> api_model_version,
        bool display_progress_bar,
        bool return_exact_agents
    ) {
        std::vector<InitializeResponse> all_responses;
    
        // Create ONE session for all requests in this function
        boost::asio::io_context ioc;
        ssl::context ctx(ssl::context::tlsv12_client);
        invertedai::Session session(ioc, ctx);
        session.set_api_key("wIvOHtKln43XBcDtLdHdXR3raX81mUE1Hp66ZRni");
        session.connect();
    
        const int num_attempts = 1 + static_cast<int>(regions.size()) / ATTEMPT_PER_NUM_REGIONS;
    
        for (size_t i = 0; i < regions.size(); ++i) {
            Region& region = regions[i];
            const Point2d region_center = region.center;
        
            // 1) Collect neighbors from other regions
            std::vector<AgentState> existing_states;
            std::vector<AgentProperties> existing_props;
            get_all_existing_agents_from_regions(regions, i, region, existing_states, existing_props);
        
            // 2) Filter neighbors by FOV of the *current* region
            std::vector<AgentState> fov_states;
            std::vector<AgentProperties> fov_props;
            fov_states.reserve(existing_states.size());
            fov_props.reserve(existing_props.size());
            for (size_t j = 0; j < existing_states.size(); ++j) {
                Point2d agent_center{existing_states[j].x, existing_states[j].y};
                if (inside_fov(region_center, region.size + AGENT_SCOPE_FOV_BUFFER, agent_center)) {
                    fov_states.push_back(existing_states[j]);
                    fov_props.push_back(existing_props[j]);
                }
            }
        
            // 3) Borrow a *small* number of nearby agents (shuffle + cap)
            constexpr size_t MAX_BORROW = 6;
            std::vector<AgentState> borrowed_states;
            std::vector<AgentProperties> borrowed_props;
            pick_borrowed_agents(fov_states, fov_props, MAX_BORROW, random_seed, borrowed_states, borrowed_props);
        
            // 4) Build inputs for this region (borrowed + native)
            std::vector<AgentState> all_agent_states = borrowed_states;
            all_agent_states.insert(all_agent_states.end(), region.agent_states.begin(), region.agent_states.end());
        
            std::vector<AgentProperties> all_agent_props = borrowed_props;
            all_agent_props.insert(all_agent_props.end(), region.agent_properties.begin(), region.agent_properties.end());
        
            if (all_agent_props.empty()) {
                continue; // nothing to initialize here
            }
        
            // 5) Retry with backoff on borrowed count, then try native-only as a last resort
            bool success = false;
            std::optional<InitializeResponse> init_res;
        
            // derive attempts budget
            int attempts_budget = std::max(2, 1 + static_cast<int>(regions.size()) / ATTEMPT_PER_NUM_REGIONS);
        
            // borrowed backoff levels: e.g., 6 -> 3 -> 1 -> 0
            std::vector<size_t> backoff { MAX_BORROW, MAX_BORROW/2, 1, 0 };
            size_t backoff_idx = 0;
        
            while (attempts_budget-- > 0 && !success) {
                try {
                    // recompute borrowed selection for the current backoff level
                    size_t take = backoff[std::min(backoff_idx, backoff.size()-1)];
                    pick_borrowed_agents(fov_states, fov_props, take, random_seed, borrowed_states, borrowed_props);
        
                    std::vector<AgentState> req_states = borrowed_states;
                    req_states.insert(req_states.end(), region.agent_states.begin(), region.agent_states.end());
        
                    std::vector<AgentProperties> req_props = borrowed_props;
                    req_props.insert(req_props.end(), region.agent_properties.begin(), region.agent_properties.end());
        
                    InitializeRequest req("{}");
                    req.set_location(location);
                    req.set_random_seed(random_seed);
        
                    if (!all_agent_states.empty()) {
                        req.set_states_history({all_agent_states});
                    } 
        
                    req.set_agent_properties(req_props);
                    req.set_location_of_interest(std::make_optional(std::make_pair(region_center.x, region_center.y)));
                    req.set_get_infractions(get_infractions);
        
                    if (traffic_light_state_history.has_value()) {
                        req.set_traffic_light_state_history(
                            convert_traffic_light_history(*traffic_light_state_history));
                    }
        
                        // Debug: inspect payload if needed
                        // std::cerr << "DEBUG InitializeRequest body:\n" << req.body_str() << "\n";
                    std::cerr << "Region " << i
                    << " -> native: " << region.agent_states.size()
                    << ", borrowed: " << borrowed_states.size() << "\n";
                    init_res = invertedai::initialize(req, &session);
                    success = true;
                } catch (const std::exception& e) {
                    std::cerr << "Region " << i << " initialize attempt failed: " << e.what() << "\n";
                    // advance backoff only if we still have borrowed agents to reduce
                    if (backoff_idx + 1 < backoff.size()) backoff_idx++;
                    continue;
                }
            }
        
            if (!success) {
                if (return_exact_agents) {
                    throw std::runtime_error("Unable to initialize region " + std::to_string(i));
                } else {
                    std::cerr << "Warning: skipping region " << i << "\n";
                    continue;
                }
            }
        
            // 6) Update region with server response
            region.agent_states     = init_res->agent_states();
            region.agent_properties = init_res->agent_properties();
            region.recurrent_states.clear();
            for (const auto& r : init_res->recurrent_states()) {
                RecurrentState rs;
                rs.packed.assign(r.begin(), r.end());
                region.recurrent_states.push_back(rs);
            }
            all_responses.push_back(*init_res);
        }
    
        return {regions, all_responses};
    }
    

  InitializeResponse consolidate_all_responses(
    const std::vector<InitializeResponse>& all_responses,
    const std::optional<std::vector<std::pair<int,int>>>& region_map,
    bool return_exact_agents,
    bool get_infractions
) {
    if (all_responses.empty()) {
          throw InvertedAIError("Unable to initialize any given region. Please check the input parameters.");
    }

    // start from a deep copy of the first response
    InitializeResponse merged = all_responses.front();

    std::vector<AgentState> agent_states;
    std::vector<AgentProperties> agent_properties;
    std::vector<std::vector<double>> recurrent_states;
    std::vector<InfractionIndicator> infractions;

    // track which agents from each response are still kept
    std::vector<std::vector<bool>> region_agent_keep_map;
    region_agent_keep_map.reserve(all_responses.size());
    for (const auto& res : all_responses) {
        region_agent_keep_map.push_back(std::vector<bool>(res.agent_properties().size(), true));
    }

    // Handle explicit region_map first
    if (region_map.has_value()) {
        for (auto [region_id, agent_id] : region_map.value()) {
            try {
                const auto& res = all_responses.at(region_id);

                agent_states.push_back(res.agent_states().at(agent_id));
                agent_properties.push_back(res.agent_properties().at(agent_id));
                recurrent_states.push_back(res.recurrent_states().at(agent_id));

                if (get_infractions && agent_id < res.infraction_indicators().size()) {
                    infractions.push_back(res.infraction_indicators().at(agent_id));
                }

                region_agent_keep_map[region_id][agent_id] = false;
            } catch (const std::out_of_range&) {
                std::string msg = "Warning: Unable to fetch specified agent ID " + std::to_string(agent_id) +
                                  " in region " + std::to_string(region_id) + ".";
                if (return_exact_agents) {
                    throw InvertedAIError(msg); // should be InvertedAIError
                } else {
                    // log debug instead of throwing if you have a logger
                }
            }
        }
    }

    // collect remaining agents from each region
    for (size_t ind = 0; ind < all_responses.size(); ++ind) {
        const auto& res = all_responses[ind];
        const auto& res_states = res.agent_states();
        const auto& res_props = res.agent_properties();
        const auto& res_recs = res.recurrent_states();

        for (size_t i = 0; i < res_states.size(); ++i) {
            if (region_agent_keep_map[ind][i]) {
                agent_states.push_back(res_states[i]);
                if (i < res_props.size()) {
                    agent_properties.push_back(res_props[i]);
                }
                if (i < res_recs.size()) {
                    recurrent_states.push_back(res_recs[i]);
                }
                if (get_infractions && i < res.infraction_indicators().size()) {
                    infractions.push_back(res.infraction_indicators()[i]);
                }
            }
        }
    }

    // overwrite merged responses fields
    merged.set_agent_states(agent_states);
    merged.set_agent_properties(agent_properties);
    merged.set_recurrent_states(recurrent_states);
    if (get_infractions) {
        merged.set_infraction_indicators(infractions);
    }

    return merged;
}

invertedai::InitializeResponse large_initialize(const invertedai::LargeInitializeConfig& cfg) {
    // validate inputs
    if(cfg.regions.size() == 0) {
        throw std::invalid_argument("At least one region must be provided.");
    }

    if(cfg.agent_properties.has_value() && cfg.agent_states.has_value() &&
       cfg.agent_properties->size() < cfg.agent_states->size()) {
        throw std::invalid_argument(
            "Invalid parameters: number of agent properties must be greater than number of agent states."
        );
    }
    // if (cfg.agent_states.has_value()) {
    //     if (!cfg.agent_properties.has_value() ||
    //         cfg.agent_properties->size() < cfg.agent_states->size()) {
    //         throw std::invalid_argument(
    //             "Invalid parameters: number of agent properties must be greater than number of agent states."
    //         );
    //     }
    // }
    // Insert agents into nearest regions (preserve indices)
    auto [regions_with_agents, region_map] = invertedai::insert_agents_into_nearest_regions(
        cfg.regions,
        cfg.agent_properties.has_value() ? cfg.agent_properties.value() : std::vector<invertedai::AgentProperties>{},
        cfg.agent_states.has_value() ? cfg.agent_states.value() : std::vector<invertedai::AgentState>{},
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
    invertedai::InitializeResponse response = invertedai::consolidate_all_responses(
        all_reponses,
        region_map,
        cfg.return_exact_agents,
        cfg.get_infractions
    );

    return (invertedai::InitializeResponse) response;
}
} // namespace invertedai