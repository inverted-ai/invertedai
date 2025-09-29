#include "large_drive.h"
#include <algorithm>
#include <iostream>
#include "invertedai/drive_response.h" // Ensure this header defines InitializeResponse
#include <random>
#include "invertedai/drive_request.h"
#include "../invertedai/api.h"
#include "invertedai/error.h"

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
    if (num_agents == 0) {
        throw InvertedAIError("Valid call must contain at least 1 agent.");
    }

    //     // --- Backward compatibility shim ---
    // if (cfg.agent_properties.empty() && cfg.agent_attributes.has_value()) {
    //     normalize_agent_properties(cfg.agent_properties, cfg.agent_attributes);
    // }

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
    Point2d region_center{ (max_x + min_x) / 2.0, (max_y + min_y) / 2.0 };

    QuadTree root(cfg.single_call_agent_limit, Region::create_square_region(region_center, region_size));

    // --- Insert all agents into quadtree
    for (int i = 0; i < num_agents; i++) {
        QuadTreeAgentInfo info{
            cfg.agent_states[i],
            cfg.agent_properties[i],
            cfg.recurrent_states.has_value()
                ? std::optional<RecurrentState>(RecurrentState(std::vector<float>(
                      cfg.recurrent_states->at(i).begin(),
                      cfg.recurrent_states->at(i).end())))
                : std::nullopt,
            i
        };
        if (!root.insert(info)) {
            throw InvertedAIError("Unable to insert agent into region.");
        }
    }

    // --- Collect all leaf nodes
    auto leaves = root.get_leaf_nodes();
    std::vector<DriveResponse> all_responses;
    std::vector<QuadTree*> non_empty_nodes;
    std::vector<int> agent_id_order;

    if (leaves.size() > 1) {
        for (auto* leaf : leaves) {
            if (leaf->get_number_of_agents_in_node() == 0) continue;

            auto region = leaf->get_regions().at(0); // assume one
            auto buffer = leaf->get_regions().at(1); // assume buffer

            // Preserve agent ID order
            for (auto& p : leaf->particles()) {
                agent_id_order.push_back(p.agent_id);
            }
                   // --- Merge region + buffer agents ---
            std::vector<AgentState> combined_states = region.agent_states;
            combined_states.insert(combined_states.end(),
                                buffer.agent_states.begin(),
                                buffer.agent_states.end());

            std::vector<AgentProperties> combined_props = region.agent_properties;
            combined_props.insert(combined_props.end(),
                                buffer.agent_properties.begin(),
                                buffer.agent_properties.end());

            std::vector<std::vector<double>> combined_recurrent;
            if (cfg.recurrent_states.has_value()) {
                combined_recurrent = region.recurrent_states;
                combined_recurrent.insert(combined_recurrent.end(),
                                        buffer.recurrent_states.begin(),
                                        buffer.recurrent_states.end());
            }

            // --- Build DriveRequest ---
            DriveRequest req("{}");
            req.set_location(cfg.location);
            req.set_get_birdview(false);
            req.set_agent_states(combined_states);
            req.set_agent_properties(combined_props);
            if (cfg.recurrent_states.has_value()) {
                req.set_recurrent_states(combined_recurrent);
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
            std::cerr << "[DEBUG] Sending DriveRequest: " << req.body_str() << std::endl;
            auto resp = drive(req, &cfg.session);
            all_responses.push_back(resp);
            non_empty_nodes.push_back(leaf);
        }

        // Flatten outputs
        std::vector<std::vector<AgentState>> states_per_leaf;
        std::vector<std::vector<std::vector<double>>> recurrent_per_leaf;
        std::vector<std::vector<bool>> inside_per_leaf;
        std::vector<std::vector<InfractionIndicator>> infractions_per_leaf;
        
        for (size_t i = 0; i < all_responses.size(); i++) {
            auto& res = all_responses[i];
            auto* leaf = non_empty_nodes[i];
            size_t n_agents = leaf->get_number_of_agents_in_node();
        
            states_per_leaf.push_back(
                std::vector<AgentState>(res.agent_states().begin(),
                                        res.agent_states().begin() + n_agents));
            recurrent_per_leaf.push_back(
                std::vector<std::vector<double>>(res.recurrent_states().begin(),
                                                 res.recurrent_states().begin() + n_agents));
            inside_per_leaf.push_back(
                std::vector<bool>(res.is_inside_supported_area().begin(),
                                  res.is_inside_supported_area().begin() + n_agents));
            if (cfg.get_infractions) {
                infractions_per_leaf.push_back(
                    std::vector<InfractionIndicator>(res.infraction_indicators().begin(),
                                                     res.infraction_indicators().begin() + n_agents));
            }
        }
        
        // --- Reorder with flatten_and_sort
        auto merged_states = flatten_and_sort(states_per_leaf, agent_id_order);
        auto merged_recurrent = flatten_and_sort(recurrent_per_leaf, agent_id_order);
        auto merged_inside = flatten_and_sort(inside_per_leaf, agent_id_order);
        std::vector<InfractionIndicator> merged_infractions;
        if (cfg.get_infractions) {
            merged_infractions = flatten_and_sort(infractions_per_leaf, agent_id_order);
        }

        DriveResponse final_resp("{}");

        // Set merged agent outputs
        final_resp.set_agent_states(merged_states);
        final_resp.set_recurrent_states(merged_recurrent);
        final_resp.set_is_inside_supported_area(merged_inside);
        
        if (cfg.get_infractions) {
            final_resp.set_infraction_indicators(merged_infractions);
        } else {
            final_resp.set_infraction_indicators({});
        }
        
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

        // Call DRIVE API synchronously (no async in C++)
        std::cerr << "[DEBUG] Sending DriveRequest: " << req.body_str() << std::endl;

        return drive(req, &cfg.session);
    }
}

} // namespace invertedai
