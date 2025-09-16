    #pragma once

    #include <vector>
    #include <map>
    #include <optional>
    #include <random>
    #include <stdexcept>
    #include "common.h"          // Region, Point, AgentType, etc.
    #include "error.h"           // InvertedAIError
    #include "invertedai/api.h"             // location_info
    #include "invertedai/data_utils.h"      // get_default_agent_properties?

    namespace invertedai {

    std::vector<Region> get_regions_default(
        const std::string& location,
        int total_num_agents,
        Session& session,
        std::optional<std::pair<float,float>> area_shape = std::nullopt,
        std::pair<float,float> map_center = {0.0f, 0.0f},
        std::optional<int> random_seed = std::nullopt,
        bool display_progress_bar = false
    );

    std::vector<invertedai::Region> get_regions_in_grid(
        float width,
        float height,
        std::pair<float,float> map_center = {0.0f,0.0f},
        float stride = 50.0f
    );

    std::vector<Region> get_number_of_agents_per_region_by_drivable_area(
        const std::string& location,
        const std::vector<Region>& regions,
        int total_num_agents,
        Session& session,
        std::optional<int> random_seed,
        bool display_progress_bar
    );
    

    } // namespace invertedai
