#include "large_drive.h"
#include <algorithm>
#include <iostream>
#include "invertedai/drive_response.h" // Ensure this header defines InitializeResponse
#include <random>
#include "invertedai/drive_request.h"
#include "../invertedai/api.h"
#include "invertedai/error.h"
#include "quadtree.h"

namespace invertedai {
// Helper: convert AgentAttributes → AgentProperties
inline AgentProperties convert_attributes_to_properties(const AgentAttributes& attrs) {
    AgentProperties props;
    if (attrs.length)           props.length = attrs.length;
    if (attrs.width)            props.width = attrs.width;
    if (attrs.rear_axis_offset) props.rear_axis_offset = attrs.rear_axis_offset;
    if (attrs.agent_type)       props.agent_type = attrs.agent_type;
    if (attrs.waypoint)         props.waypoint = attrs.waypoint;
    return props;
}

// Shim: normalize cfg.agent_properties (convert any AgentAttributes to AgentProperties)
inline void normalize_agent_properties(std::vector<AgentProperties>& agent_properties,
                                       const std::optional<std::vector<AgentAttributes>>& maybe_attrs) {
    bool used_attributes = false;

    if (maybe_attrs.has_value()) {
        const auto& attrs_vec = maybe_attrs.value();
        // Convert every AgentAttributes into AgentProperties
        for (const auto& attrs : attrs_vec) {
            agent_properties.push_back(convert_attributes_to_properties(attrs));
        }
        used_attributes = true;
    }

    if (used_attributes) {
        std::cerr << "[WARN] agent_attributes is deprecated. Please use agent_properties instead.\n";
    }
}

DriveResponse large_drive(LargeDriveConfig& cfg) {
    
    int num_agents = static_cast<int>(cfg.agent_states.size());
    if (num_agents <= 0) {
        throw InvertedAIError("Valid call must contain at least 1 agent.");
    }

    // STRICT size checks (fixes segfaults on cfg.agent_properties[i])
    const auto props_sz = static_cast<int>(cfg.agent_properties.size());
    if (props_sz != num_agents) {
        std::ostringstream oss;
        oss << "agent_states.size()=" << num_agents
            << " but agent_properties.size()=" << props_sz
            << " (they MUST match)";
        throw InvertedAIError(oss.str());
    }

    if (cfg.recurrent_states.has_value()) {
        const auto rec_sz = static_cast<int>(cfg.recurrent_states->size());
        if (rec_sz != num_agents) {
            std::ostringstream oss;
            oss << "agent_states.size()=" << num_agents
                << " but recurrent_states.size()=" << rec_sz
                << " (they MUST match when provided)";
            throw InvertedAIError(oss.str());
        }
    }

    if (num_agents != static_cast<int>(cfg.agent_properties.size())) {
        if (cfg.recurrent_states.has_value() &&
            num_agents != static_cast<int>(cfg.recurrent_states->size())) {
            throw InvertedAIError("Input lists are not of equal size.");
        }
    }

    if (cfg.single_call_agent_limit > DRIVE_MAXIMUM_NUM_AGENTS) {
        std::cerr << "[WARN] single_call_agent_limit capped at "
                  << DRIVE_MAXIMUM_NUM_AGENTS << "\n";
        cfg.single_call_agent_limit = DRIVE_MAXIMUM_NUM_AGENTS;
    }

    // --- Compute root region bounds
    double max_x = -1e9, min_x = 1e9, max_y = -1e9, min_y = 1e9;
    for (const auto& s : cfg.agent_states) {
        max_x = std::max(max_x, s.x);
        min_x = std::min(min_x, s.x);
        max_y = std::max(max_y, s.y);
        min_y = std::min(min_y, s.y);
    }
    double region_size = std::ceil(std::max(max_x - min_x, max_y - min_y)) + QUADTREE_SIZE_BUFFER;
    Point2d region_center{
        std::round((max_x + min_x) / 2.0),
        std::round((max_y + min_y) / 2.0)
    };
    
   
        std::unique_ptr<QuadTree> root;
        try {
            root = std::make_unique<QuadTree>(
                cfg.single_call_agent_limit,
                Region::create_square_region(region_center, region_size)
            );
            std::cerr << "[CHECKPOINT] QuadTree constructed OK\n";
        } catch (const std::exception& e) {
            std::cerr << "[FATAL] QuadTree ctor threw: " << e.what()
                    << " (center=" << region_center.x << "," << region_center.y
                    << " size=" << region_size << " cap=" << cfg.single_call_agent_limit << ")\n";
            throw;
        }
    // --- Insert all agents into quadtree
    for (int i = 0; i < num_agents; i++) {
        QuadTreeAgentInfo info{
            cfg.agent_states[i],                                // AgentState
            cfg.recurrent_states.has_value()
                ? std::optional<std::vector<double>>(cfg.recurrent_states->at(i))
                : std::nullopt,                                 // recurrent_state
            cfg.agent_properties[i],                            // AgentProperties
            i                                                   // agent_id
        };
        if (!root->insert(info)) {
            throw InvertedAIError("Unable to insert agent into region.");
        }
    }
    std::cout << "finished inserting" << std::endl;
    auto leaves = root->get_leaf_nodes(); 
    std::cout << "3" << std::endl;
    // --- Collect all leaf nodes
    if (leaves.size() > 1) {
        std::vector<DriveResponse> all_responses;
        std::vector<QuadTree*> non_empty_nodes;
        std::vector<int> agent_id_order;
    
        for (auto* leaf : leaves) {
            // get core region and buffer from leaf
            const Region& core   = leaf->region();         // must exist in your QuadTree API
            const Region& buffer = leaf->region_buffer();  // ditto (neighbors visible set)
            // particles (original agent ids for core region)
            std::vector<int> region_agent_ids;
            region_agent_ids.reserve(leaf->particles().size());
            for (const auto& p : leaf->particles()) {
                region_agent_ids.push_back(p.agent_id);
            }
            // region_agent_ids.reserve(leaf->particles().size());
            // for (size_t p = 0; p < leaf->particles().size(); p++) {
            //     region_agent_ids.push_back(leaf->particles()[p].agent_id);
            // }
            
            if (!core.agent_states.empty()) {
                non_empty_nodes.push_back(leaf);
                agent_id_order.insert(agent_id_order.end(),
                                      region_agent_ids.begin(),
                                      region_agent_ids.end());
            
                // Build a request for REGION + BUFFER (Python concatenates region then buffer in that order)
                DriveRequest req("{}");
                req.set_location(cfg.location);
                req.set_get_birdview(false);
                // states
                {
                    std::vector<AgentState> s = core.agent_states;
                    s.insert(s.end(), buffer.agent_states.begin(), buffer.agent_states.end());
                    req.set_agent_states(s);
                }
                // properties
                {
                    std::vector<AgentProperties> p = core.agent_properties;
                    p.insert(p.end(), buffer.agent_properties.begin(), buffer.agent_properties.end());
                    req.set_agent_properties(p);
                }
                // recurrent (optional)
                if (cfg.recurrent_states.has_value()) {
                    std::vector<std::vector<double>> r = core.recurrent_states;
                    r.insert(r.end(), buffer.recurrent_states.begin(), buffer.recurrent_states.end());
                    req.set_recurrent_states(r);
                }
                if (cfg.traffic_lights_states.has_value()) req.set_traffic_lights_states(cfg.traffic_lights_states.value());
                if (cfg.light_recurrent_states.has_value()) req.set_light_recurrent_states(cfg.light_recurrent_states.value());
                req.set_get_infractions(cfg.get_infractions);
                if (cfg.random_seed.has_value()) req.set_random_seed(cfg.random_seed.value());
                if (cfg.api_model_version.has_value()) req.set_model_version(cfg.api_model_version.value());
    
                // Sync call (Python can do async; C++ is sync)
                all_responses.push_back(drive(req, &cfg.session));
            }
        }
        // Flatten outputs
        std::vector<std::vector<AgentState>> states_per_leaf;
        std::vector<std::vector<std::vector<double>>> recurrent_per_leaf;
        std::vector<std::vector<bool>> inside_per_leaf;
        std::vector<std::vector<InfractionIndicator>> infractions_per_leaf;
        
        for (size_t i = 0; i < all_responses.size(); i++) {
            auto& res  = all_responses[i];
            auto* leaf = non_empty_nodes[i];
        
            if (!leaf) {
                std::cerr << "[CRASH] leaf pointer null at i=" << i << "\n";
                continue;
            }
        
            size_t n_agents = leaf->get_number_of_agents_in_node();
            n_agents = std::min(n_agents, res.agent_states().size());
            n_agents = std::min(n_agents, res.recurrent_states().size());
        
            // Check sizes before slicing
            std::cerr << " res.agent_states=" << res.agent_states().size()
                      << " recurrent=" << res.recurrent_states().size()
                      << " inside=" << res.is_inside_supported_area().size()
                      << " infractions=" << res.infraction_indicators().size()
                      << "\n";
        
            // Early continue to isolate
            if (res.agent_states().empty()) {
                std::cerr << "[LEAF " << i << "] agent_states empty, skipping.\n";
                continue;
            }
        
            //  never slice beyond what API returned
            const auto states_vec    = res.agent_states();
            const auto recurrent_vec = res.recurrent_states();
            const auto inside_vec    = res.is_inside_supported_area();
            const auto infr_vec      = res.infraction_indicators();
            
            // size_t coreN = non_empty_nodes[i]->get_number_of_agents_in_node();
            const size_t coreN = leaf->particles().size();
            const auto& s = res.agent_states();
            const auto& r = res.recurrent_states();
            const auto& in = res.is_inside_supported_area();
            const auto& inf = res.infraction_indicators();
    
            states_per_leaf.emplace_back(s.begin(), s.begin() + std::min(coreN, s.size()));
            recurrent_per_leaf.emplace_back(r.begin(), r.begin() + std::min(coreN, r.size()));
            inside_per_leaf.emplace_back(in.begin(), in.begin() + std::min(coreN, in.size()));
            if (cfg.get_infractions) {
                infractions_per_leaf.emplace_back(inf.begin(), inf.begin() + std::min(coreN, inf.size()));
            }
        }            
        //     size_t ns = std::min(coreN, states_vec.size());
        //     size_t nr = std::min(coreN, recurrent_vec.size());
        //     size_t ni = std::min(coreN, inside_vec.size());
        //     size_t nf = cfg.get_infractions ? std::min(coreN, infr_vec.size()) : 0;
            
        //     std::cerr << "[DEBUG] Leaf " << i
        //               << " core=" << coreN
        //               << " ns=" << ns << " nr=" << nr << " ni=" << ni << " nf=" << nf
        //               << " (resp sizes: " << states_vec.size() << ", "
        //                                   << recurrent_vec.size() << ", "
        //                                   << inside_vec.size() << ", "
        //                                   << (cfg.get_infractions ? infr_vec.size() : 0) << ")\n";
            
        //     states_per_leaf.emplace_back(states_vec.begin(), states_vec.begin() + ns);
        //     recurrent_per_leaf.emplace_back(recurrent_vec.begin(), recurrent_vec.begin() + nr);
        //     inside_per_leaf.emplace_back(inside_vec.begin(), inside_vec.begin() + ni);
        //     if (cfg.get_infractions) {
        //         infractions_per_leaf.emplace_back(infr_vec.begin(), infr_vec.begin() + nf);
        //     }
        // }
        size_t flat_states = 0;
        for (const auto& v : states_per_leaf) flat_states += v.size();
        
        std::cerr << "[DEBUG] agent_id_order.size=" << agent_id_order.size()
                  << " flattened total=" << flat_states << "\n";
        
        if (agent_id_order.size() != flat_states) {
            throw std::runtime_error("index_list vs flattened data mismatch — check quadtree core assignment");
        }
        // --- Reorder with flatten_and_sort
        std::cerr << "[CHECKPOINT] About to flatten states\n";
        auto merged_states = flatten_and_sort(states_per_leaf, agent_id_order);
        
        std::cerr << "[CHECKPOINT] Flatten states done, flatten recurrent...\n";
        auto merged_recurrent = flatten_and_sort(recurrent_per_leaf, agent_id_order);
        
        std::cerr << "[CHECKPOINT] Flatten recurrent done, flatten inside...\n";
        auto merged_inside = flatten_and_sort(inside_per_leaf, agent_id_order);
        
        std::cerr << "[CHECKPOINT] Flatten inside done\n";
        std::vector<InfractionIndicator> merged_infractions;
        if (cfg.get_infractions) {
            merged_infractions = flatten_and_sort(infractions_per_leaf, agent_id_order);
        }

        DriveResponse final_resp("{}");

        // Set merged agent outputs
        final_resp.set_agent_states(merged_states);
        final_resp.set_recurrent_states(merged_recurrent);
        final_resp.set_is_inside_supported_area(merged_inside);
        final_resp.set_birdview({});
        
        if (cfg.get_infractions) {
            final_resp.set_infraction_indicators(merged_infractions);
        } else {
            final_resp.set_infraction_indicators({});
        }
        if (all_responses[0].traffic_lights_states().has_value()) {
            for (auto& kv : *all_responses[0].traffic_lights_states()) {
                if (kv.second.empty()) {
                    std::cerr << "[WARN] traffic_lights_states has empty value for key=" << kv.first << std::endl;
                }
            }
        }
        // if (all_responses[0].light_recurrent_states().has_value()) {
        //     for (LightRecurrentState kv : *all_responses[0].light_recurrent_states()) {
        //             if (kv.state.empty()) {
        //                 std::cerr << "[WARN] light_recurrent_states has empty value for key=" << kv.state << std::endl;
        //             }
        //             if (kv.time_remaining.empty()) {
        //                 std::cerr << "[WARN] light_recurrent_states has empty value for key=" << kv.state << std::endl;
        //             }
        //     }
        // }
        
        // if (all_responses[0].model_version().has_value()) {
        //     std::cerr << "model_version=" << *all_responses[0].model_version() << std::endl;
        // } else {
        //     std::cerr << "model_version is null" << std::endl;
        // }
        // Copy over shared fields from the first response
        if (!all_responses.empty()) {
            final_resp.set_traffic_lights_states(
                all_responses[0].traffic_lights_states().value_or(std::map<std::string,std::string>())
            );
            if (all_responses[0].light_recurrent_states().has_value()) {
                final_resp.set_light_recurrent_states(all_responses[0].light_recurrent_states().value());
            }
            final_resp.set_birdview({}); // always None/empty in large_drive
            final_resp.set_infraction_indicators(
                cfg.get_infractions ? merged_infractions : std::vector<InfractionIndicator>{}
            );
        }
        return final_resp;

    } else {
        // --- If no subdivision, call DRIVE directly
        DriveRequest req("{}");
        std::cout << "hi\n";
        req.set_location(cfg.location);
        std::cerr << "[DEBUG] Location just set to: " << cfg.location << std::endl;
        std::cout << "ho\n";
        req.set_get_birdview(false);
        req.set_agent_states(cfg.agent_states);
        req.set_agent_properties(cfg.agent_properties);
        if (cfg.recurrent_states.has_value()) {
            req.set_recurrent_states(cfg.recurrent_states.value());
        }
        if (cfg.traffic_lights_states.has_value()) {
            req.set_traffic_lights_states(cfg.traffic_lights_states.value());
        }
        if (cfg.light_recurrent_states.has_value()) {
            req.set_light_recurrent_states(cfg.light_recurrent_states.value());
        }
        req.set_get_infractions(cfg.get_infractions);
        if (cfg.random_seed.has_value()) {
            req.set_random_seed(cfg.random_seed.value());
        }
        
        if (cfg.api_model_version.has_value()) {
            req.set_model_version(cfg.api_model_version.value());
        }

        // Call DRIVE API synchronously (no async in C++)
        try {
            std::string body = req.body_str();
            cfg.logger.append_request(body, "drive");
            DriveResponse res = drive(req, &cfg.session);
            std::cerr << "[API DEBUG] Response sizes:\n"
          << "  agent_states=" << res.agent_states().size() << "\n"
          << "  recurrent_states=" << res.recurrent_states().size() << "\n"
          << "  is_inside_supported_area=" << res.is_inside_supported_area().size() << "\n"
          << "  infractions=" << res.infraction_indicators().size() << "\n"
          << "  light_recurrent_states=" << (res.light_recurrent_states().has_value() 
                                               ? res.light_recurrent_states()->size() : 0) << "\n"
          << "  traffic_lights_states=" << (res.traffic_lights_states().has_value() 
                                               ? res.traffic_lights_states()->size() : 0) << "\n";
            return res;
        } catch (const std::exception& e) {
            std::cerr << "[FATAL] Exception while serializing DriveRequest: " << e.what() << std::endl;
            throw;
        }
        
    }
}

LargeDriveWithRegions large_drive_with_regions(LargeDriveConfig& cfg) {
    int num_agents = static_cast<int>(cfg.agent_states.size());
    if (num_agents == 0) {
        throw InvertedAIError("Valid call must contain at least 1 agent.");
    }

    // --- Compute root region bounds ---
    double max_x = -1e9, min_x = 1e9, max_y = -1e9, min_y = 1e9;
    for (const auto& s : cfg.agent_states) {
        max_x = std::max(max_x, s.x);
        min_x = std::min(min_x, s.x);
        max_y = std::max(max_y, s.y);
        min_y = std::min(min_y, s.y);
    }
    double region_size = std::ceil(std::max(max_x - min_x, max_y - min_y)) + QUADTREE_SIZE_BUFFER;
    Point2d region_center{
        std::round((max_x + min_x) / 2.0),
        std::round((max_y + min_y) / 2.0)
    };

    QuadTree root(cfg.single_call_agent_limit, Region::create_square_region(region_center, region_size));

    // --- Insert all agents ---
    for (int i = 0; i < num_agents; i++) {
        QuadTreeAgentInfo info{ cfg.agent_states[i],
                                cfg.recurrent_states ? std::optional<std::vector<double>>(cfg.recurrent_states->at(i)) : std::nullopt,
                                cfg.agent_properties[i],
                                i };
        if (!root.insert(info)) {
            throw InvertedAIError("Unable to insert agent into region.");
        }
    }

    // --- Get leaf regions for visualization ---
    std::vector<Region> leaf_regions = root.get_regions();

    // --- Call original drive logic (using your existing large_drive) ---
    DriveResponse resp = large_drive(cfg);

    return { std::move(resp), std::move(leaf_regions) };
}

} // namespace invertedai
