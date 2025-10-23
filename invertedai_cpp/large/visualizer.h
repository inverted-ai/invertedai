#include <opencv2/opencv.hpp>
#include "invertedai/session.h"
#include "invertedai/location_info_response.h"
#include "invertedai/drive_response.h"
#include "invertedai/initialize_response.h"
#include "large_initialize/large_initialize.h"
#include "large_drive/large_drive.h"
#include "large/visualizer_helpers.h"


namespace invertedai {

// High-level visualization entry points


/**
 * @brief Render and save visualization for large_initialize() output.
 *
 * Produces a stitched map with initialized agents overlaid.
 */
void visualize_large_initialize(
    const std::string& location,
    Session& session,
    const std::vector<invertedai::Region> final_regions,
    const std::vector<Region>& all_tiles,
    const LocationInfoResponse& li_res,
    bool flip_x
);
void visualize_large_drive(
    const LargeDriveConfig& drive_cfg,
    const std::vector<invertedai::Region> leaf_regions,
    const std::vector<invertedai::Region> final_regions,
    const LocationInfoResponse& li_res,
    const std::optional<std::map<std::string, std::string>>& traffic_lights_states,
    const std::vector<Region>& drive_tiles,
    const std::unordered_map<std::pair<double,double>, cv::Mat, invertedai::PairHash>& drive_cached_tiles,

    cv::VideoWriter writer,
    bool flip_x,
    int step
);
void visualize_large_drive_v2(
    const LargeDriveConfig& drive_cfg,
    const std::vector<invertedai::Region> final_regions,
    const LocationInfoResponse& li_res,
    bool flip_x,
    int sim_length
);

}  // namespace invertedai