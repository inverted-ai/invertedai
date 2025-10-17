#include "large_drive.h"
#include <algorithm>
#include <iostream>
#include "invertedai/drive_response.h" // Ensure this header defines InitializeResponse
#include <random>
#include "invertedai/drive_request.h"
#include "../invertedai/api.h"
#include "invertedai/error.h"
#include "quadtree.h"

// for async driving
#include <future>
#include <mutex>

namespace invertedai {

/**
 * @brief Perform a large-scale DRIVE simulation with automatic spatial subdivision.
 *
 * This function partitions the full set of agents into multiple quadtree regions
 * when the population exceeds the single-call capacity. Each region
 * corresponds to a `DriveRequest` to the InvertedAI API, and can be executed
 * either synchronously or asynchronously depending on `cfg.async_api_calls`.
 *
 *  Algorithm Overview
 * 1. Input validation:  
 *    Ensures that all agent vectors (`states`, `properties`, and optionally
 *    `recurrent_states`) are non-empty and consistent in size.
 *
 * 2. Quadtree construction:  
 *    Computes the world-space bounding box from agent positions and constructs
 *    a quadtree (`QuadTree`) where each leaf contains at most
 *    `cfg.single_call_agent_limit` agents. If exceeded, the region subdivides
 *    recursively until capacity is satisfied.
 *
 * 3. Request generation:  
 *    For each non-empty leaf node, a `DriveRequest` is created containing:
 *    - Agent states and properties for the core region and its spatial buffer.
 *    - Optional recurrent states and traffic light data.
 *    - Configuration flags (`get_infractions`, `api_model_version`, etc.).
 *
 * 4. Parallel execution:  
 *    If `cfg.async_api_calls` is true, each leaf node launches its own
 *    asynchronous DRIVE API call using a local session. Otherwise requests
 *    are executed sequentially.
 *
 * 5. Result aggregation:  
 *    Once all responses are collected:
 *    - Agent states, recurrent states, and infractions are flattened and
 *      re-sorted by agent ID order.
 *    - Any size mismatches are clamped to the smallest valid set.
 *    - The first non-empty response contributes traffic light and recurrent data.
 *
 * 6. Final response:  
 *    Returns a unified `DriveResponse` object containing merged outputs
 *    across all quadtree regions.
 *
 *  Parameters
 * @param cfg Configuration object containing all simulation parameters
 * (see `LargeDriveConfig`).
 *
 *  Returns
 * A combined `DriveResponse` with updated agent states, recurrent states,
 * infractions, and traffic light.
 *
 *  Throws
 * - `InvertedAIError` if agent data are invalid or DRIVE API requests fail.
 * - `std::runtime_error` for unexpected internal inconsistencies.
 *
 */
DriveResponse large_drive(LargeDriveConfig& cfg) {
    // size checks
    int num_agents = static_cast<int>(cfg.agent_states.size());
    int props_size  = static_cast<int>(cfg.agent_properties.size());
    if (num_agents <= 0) {
        throw InvertedAIError("valid call must contain at least 1 agent.");
    }
    if (props_size != num_agents) {
        std::string msg = 
            "agent_states.size()=" + std::to_string(num_agents) +
            " but agent_properties.size()=" + std::to_string(props_size) +
            " (they must match)";
        throw InvertedAIError(msg);
    }
    
    if (cfg.recurrent_states.has_value()) {
        const auto rec_size = static_cast<int>(cfg.recurrent_states->size());
        if (rec_size != num_agents) {
            std::string msg =
                "agent_states.size()=" + std::to_string(num_agents) +
                " but recurrent_states.size()=" + std::to_string(rec_size) +
                " (they must match when provided)";
            throw InvertedAIError(msg);
        }
    }

    if (num_agents != static_cast<int>(cfg.agent_properties.size())) {
        if (cfg.recurrent_states.has_value() &&
            num_agents != static_cast<int>(cfg.recurrent_states->size())) {
            throw InvertedAIError("Mismatch between Agent_states, Agent_properties, and Recurrent_states sizes.");
        }
    }

    if (cfg.single_call_agent_limit > DRIVE_MAXIMUM_NUM_AGENTS) {
        std::cerr << "[WARNING] single_call_agent_limit capped at "
                  << DRIVE_MAXIMUM_NUM_AGENTS << "\n";
        cfg.single_call_agent_limit = DRIVE_MAXIMUM_NUM_AGENTS;
    }

    // compute root region bounds
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
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] QuadTree constructor threw: " << e.what()
                << " (region center=" << region_center.x << "," << region_center.y
                << " region size=" << region_size << " capacity=" << cfg.single_call_agent_limit << ")\n";
        throw;
    }
    // insert all agents into quadtree
    for (int i = 0; i < num_agents; i++) {
        QuadTreeAgentInfo info{
            cfg.agent_states[i],                                
            cfg.recurrent_states.has_value()
                ? std::optional<std::vector<double>>(cfg.recurrent_states->at(i))
                : std::nullopt,  
            cfg.agent_properties[i], 
            i  
        };
        if (!root->insert(info)) {
            throw InvertedAIError("unable to insert agent into region.");
        }
    }
    auto leaves = root->get_leaf_nodes(); 
    // collect all leaf nodes
    if (leaves.size() > 1) {
        std::vector<DriveResponse> all_responses;
        std::vector<QuadTree*> non_empty_nodes;
        std::vector<int> agent_id_order;

        std::vector<LeafTask> tasks;
        tasks.reserve(leaves.size());

        for (auto* leaf : leaves) {
            // get core region and buffer from leaf
            const Region& core   = leaf->region();         
            const Region& buffer = leaf->region_buffer(); 
            // particles 
            std::vector<int> region_agent_ids;
            region_agent_ids.reserve(leaf->particles().size());
            for (const auto& p : leaf->particles()) {
                region_agent_ids.push_back(p.agent_id);
            }
            if (!core.agent_states.empty()) {
                non_empty_nodes.push_back(leaf);
                agent_id_order.insert(agent_id_order.end(),
                                      region_agent_ids.begin(),
                                      region_agent_ids.end());
            
                // region and buffer
                DriveRequest req("{}");
                req.set_location(cfg.location);
                req.set_get_birdview(false);
                // states
                const size_t MAX_NUM_AGENTS = cfg.single_call_agent_limit;

                // building states with clamp
                std::vector<AgentState> s;
                for (auto &st : core.agent_states) {
                    if (s.size() >= MAX_NUM_AGENTS) break;
                    s.push_back(st);
                }
                for (auto &st : buffer.agent_states) {
                    if (s.size() >= MAX_NUM_AGENTS) break;
                    s.push_back(st);
                }
                req.set_agent_states(s);
                
                // properties
                std::vector<AgentProperties> p;
                for (auto &prop : core.agent_properties) {
                    if (p.size() >= MAX_NUM_AGENTS) break;
                    p.push_back(prop);
                }
                for (auto &prop : buffer.agent_properties) {
                    if (p.size() >= MAX_NUM_AGENTS) break;
                    p.push_back(prop);
                }
                req.set_agent_properties(p);
                
                // recurrent
                if (cfg.recurrent_states.has_value()) {
                    std::vector<std::vector<double>> r;
                    for (auto &rec : core.recurrent_states) {
                        if (r.size() >= MAX_NUM_AGENTS) break;
                        r.push_back(rec);
                    }
                    for (auto &rec : buffer.recurrent_states) {
                        if (r.size() >= MAX_NUM_AGENTS) break;
                        r.push_back(rec);
                    }
                    req.set_recurrent_states(r);
                }                
                if (cfg.traffic_lights_states.has_value()) req.set_traffic_lights_states(cfg.traffic_lights_states.value());
                if (cfg.light_recurrent_states.has_value()) req.set_light_recurrent_states(cfg.light_recurrent_states.value());

                req.set_get_infractions(cfg.get_infractions);
                if (cfg.random_seed.has_value()) req.set_random_seed(cfg.random_seed.value());
                if (cfg.api_model_version.has_value()) req.set_model_version(cfg.api_model_version.value());

                if(cfg.async_api_calls) { // check if async calls is requested
                    // async call
                    tasks.push_back(LeafTask{ non_empty_nodes.size() - 1, std::move(req) });
                } else {
                    // sync call
                    DriveResponse r = drive(req, &cfg.session);
                    all_responses.push_back(std::move(r));
                }
            }
        }
        all_responses.resize(non_empty_nodes.size());

        std::vector<std::future<std::pair<size_t, DriveResponse>>> futs;
        futs.reserve(tasks.size());
        if(cfg.async_api_calls) {
        std::cout << "Launching " << tasks.size() << " async drive tasks\n";
        
            for (auto &t : tasks) {
                if(!cfg.async_api_calls) break; // safety
                futs.emplace_back(std::async(std::launch::async, [&cfg](LeafTask lt){
                    try {
                        net::io_context ioc;
                        ssl::context ctx(ssl::context::sslv23_client);
                        invertedai::Session local_sess(ioc, ctx);
                        local_sess.set_api_key(cfg.api_key);
                        local_sess.connect();
                        DriveResponse r = drive(lt.req, &local_sess);
                        return std::pair<size_t, DriveResponse>{ lt.idx, std::move(r) };
                    } catch (const std::exception &e) {
                        std::cerr << "[ASYNC] Leaf " << lt.idx 
                                << " exception: " << e.what() << "\n";
                        throw;
                    }
                }, t));
            }
            // collect all async results
            for (auto &f : futs) {
                auto [idx, resp] = f.get();
                all_responses[idx] = std::move(resp);
            }
        } else {
            for (auto &t : tasks) {
                DriveResponse r = drive(t.req, &cfg.session);
                all_responses[t.idx] = std::move(r);  
            }
        }
        // flatten outputs
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
        
            if (res.agent_states().empty()) {
                continue;
            }
        
            //  use API returned vectors
            const auto states_vec    = res.agent_states();
            const auto recurrent_vec = res.recurrent_states();
            const auto inside_vec    = res.is_inside_supported_area();
            const auto infr_vec      = res.infraction_indicators();
            
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
        auto merged_states = flatten_and_sort(states_per_leaf, agent_id_order);
        auto merged_recurrent = flatten_and_sort(recurrent_per_leaf, agent_id_order);
        auto merged_inside = flatten_and_sort(inside_per_leaf, agent_id_order);

        if (merged_states.size() != merged_recurrent.size()) {
            std::cerr << "[WARN] Mismatch in merged agents: states=" << merged_states.size()
                      << " recurrent=" << merged_recurrent.size()
                      << " — continuing with min-size subset." << std::endl;
            const size_t n = std::min(merged_states.size(), merged_recurrent.size());
            merged_states.resize(n);
            merged_recurrent.resize(n);
        }
        for (int i = 0; i < merged_recurrent.size(); ++i) {
            if (merged_recurrent[i].size() == 0) {
                std::cout << "[WARN] empty recurrent at index " << i << std::endl;
            }
        }
        std::vector<InfractionIndicator> merged_infractions;
        if (cfg.get_infractions) {
            merged_infractions = flatten_and_sort(infractions_per_leaf, agent_id_order);
        }
        DriveResponse final_resp("{}");
        final_resp.set_agent_states(merged_states);
        final_resp.set_recurrent_states(merged_recurrent);
        final_resp.set_is_inside_supported_area(merged_inside);

        final_resp.set_birdview({});
        
        if (cfg.get_infractions) {
            final_resp.set_infraction_indicators(merged_infractions);
        } else {
            final_resp.set_infraction_indicators({});
        }
        
        // old debugging code - might delete ?
        auto tls_opt = all_responses[0].traffic_lights_states();
        if (tls_opt && !tls_opt->empty()) {
            for (const auto& kv : *tls_opt) {
                final_resp.set_traffic_lights_states(*tls_opt);
                if (kv.second.empty()) {
                    std::cerr << "[WARN] traffic_lights_states has empty value for key=" << kv.first << std::endl;
                }
            }
        }
        
        if (!all_responses.empty()) {
            try {
                final_resp.set_traffic_lights_states(
                    all_responses[0].traffic_lights_states().value_or(std::map<std::string,std::string>())
                );

                if (all_responses[0].light_recurrent_states().has_value()) {
                    final_resp.set_light_recurrent_states(all_responses[0].light_recurrent_states().value());
                }
                final_resp.set_birdview({});
                final_resp.set_infraction_indicators(
                    cfg.get_infractions ? merged_infractions : std::vector<InfractionIndicator>{}
                );

            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Exception inside final_resp DriveRequest: " << e.what() << std::endl;
                throw;
            }
        }
        return final_resp;

    } else {
        // if no subdivision, call drive directly
        DriveRequest req("{}");
        req.set_location(cfg.location);
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

        // call drive api synchronously 
        try {
            std::string body = req.body_str();
            cfg.logger.append_request(body, "drive");
            DriveResponse res = drive(req, &cfg.session);
            return res;
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception while serializing DriveRequest: " << e.what() << std::endl;
            throw;
        }
        
    }
}

// function to visualize the leaf nodes
LargeDriveWithRegions large_drive_with_regions(LargeDriveConfig& cfg) {
    int num_agents = static_cast<int>(cfg.agent_states.size());
    if (num_agents == 0) {
        throw InvertedAIError("Valid call must contain at least 1 agent.");
    }

    // compute root region bounds
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

    for (int i = 0; i < num_agents; i++) {
        QuadTreeAgentInfo info{ cfg.agent_states[i],
                                cfg.recurrent_states ? std::optional<std::vector<double>>(cfg.recurrent_states->at(i)) : std::nullopt,
                                cfg.agent_properties[i],
                                i };
        if (!root.insert(info)) {
            throw InvertedAIError("Unable to insert agent into region.");
        }
    }

    std::vector<Region> leaf_regions = root.get_regions();

    DriveResponse resp = large_drive(cfg);

    return { std::move(resp), std::move(leaf_regions) };
}

} // namespace invertedai
