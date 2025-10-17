#ifndef INVERTEDAI_QUADTREE_H
#define INVERTEDAI_QUADTREE_H
/**
 * @file quadtree.h
 * @brief Spatial partitioning structure for large-scale DRIVE and INITIALIZE simulations.
 *
 * Provides a recursive quadtree implementation that subdivides the simulation
 * space into smaller regions, ensuring each region contains no more than a
 * configured number of agents. Each leaf node maintains its own agent data,
 * including dynamic states, static properties, and recurrent memory vectors.
 */

#include <vector>
#include <optional>
#include "common.h"
#include "invertedai/data_utils.h"    
namespace invertedai {
/**
 * @brief Field of view (FOV) buffer radius in meters.
 *
 * Agents within this distance from a region’s boundary are shared between
 * neighboring regions to ensure continuity during simulation updates.
 */
constexpr double BUFFER_FOV = 35.0;
/**
 * @brief Additional spatial buffer added when calculating the quadtree’s root region size.
 */
constexpr double QUADTREE_SIZE_BUFFER = 1.0;


/**
 * @brief Represents a single agent's data stored in a quadtree node.
 *
 * Combines dynamic state, static properties, recurrent memory, and an ID index
 * into one structure. Used as the fundamental particle unit within each quadtree node.
 */
struct QuadTreeAgentInfo {
    AgentState agent_state;
    std::optional<std::vector<double>> recurrent_state;
    AgentProperties agent_properties;
    int agent_id;
};

/**
 * @brief Hierarchical spatial partitioning structure for agent distribution.
 *
 * Each node represents a square region in 2D world coordinates. If the number
 * of contained agents exceeds the node’s capacity, it subdivides into four
 * child nodes (north-west, north-east, south-west, south-east).
 *
 * Used by `large_initialize()` and `large_drive()` to split the world into
 * manageable DRIVE or INITIALIZE API calls, ensuring performance and compliance
 * with the maximum agent limit per call.
 */
class QuadTree {
public:
    /**
     * @brief Construct a new QuadTree node.
     *
     * @param capacity Maximum number of agents allowed per leaf node before subdivision.
     * @param region Spatial bounds (square) of this node.
     */
    QuadTree(int capacity, const Region& region);

    /**
     * @brief Insert an agent into the quadtree.
     *
     * Automatically subdivides if the node exceeds capacity.
     *
     * @param particle Agent data (state, properties, recurrent, id).
     * @param is_particle_placed Whether this agent was previously placed in another node (used internally).
     * @return True if the insertion succeeded.
     */
    bool insert(const QuadTreeAgentInfo& particle, bool is_particle_placed = false);

    /**
     * @brief Get all region geometries from the tree (for visualization/debugging).
     *
     * @return Vector of regions corresponding to all quadtree leaves.
     */
    std::vector<invertedai::Region> get_regions() const;

    /**
     * @brief Retrieve pointers to all leaf nodes in the quadtree.
     *
     * Useful for iterating over simulation regions or constructing DRIVE requests.
     *
     * @return Vector of raw pointers to leaf nodes.
     */
    std::vector<QuadTree*> get_leaf_nodes();

    /**
     * @brief Get the number of agents in this node (excluding children).
     */
    size_t get_number_of_agents_in_node() const;

    /**
     * @brief Get the agents stored in this region's node.
     */
    const std::vector<QuadTreeAgentInfo>& particles() const { return particles_; }

        /**
     * @brief Get the outside agents overlapping this region's FOV due to BUFFER_FOV.
     */
    const std::vector<QuadTreeAgentInfo>& particles_buffer() const { return particles_buffer_; }


    /**
     * @brief Get this node’s region (without buffer).
     */
    const Region& region() const { return region_; }

        /**
     * @brief Get this node’s buffered region (expanded by BUFFER_FOV).
     */
    const Region& region_buffer() const { return region_buffer_; }

private:
    int capacity_;                         // Maximum number of agents allowed before subdivision
    invertedai::Region region_;            // Spatial bounds of the current node
    Region region_buffer_;                 // Region expanded by BUFFER_FOV for overlapping neighbors
    bool leaf_;                            // True if this node has no children

    std::vector<QuadTreeAgentInfo> particles_;         // Agents contained in this node
    std::vector<QuadTreeAgentInfo> particles_buffer_;  // Agents near boundaries (visible to neighbors)

    std::unique_ptr<QuadTree> north_west;  // Pointer to the north-west child
    std::unique_ptr<QuadTree> north_east;  // Pointer to the north-east child
    std::unique_ptr<QuadTree> south_west;  // Pointer to the south-west child
    std::unique_ptr<QuadTree> south_east;  // Pointer to the south-east child

    /**
     * @brief Split this quadtree node into four equal child nodes.
     *
     * Each child covers one quadrant of the parent region. All agents in this node
     * are reinserted into the appropriate children. The parent becomes an internal
     * node and no longer stores agents directly.
     *
     * Called automatically when a node exceeds its capacity.
     */
    void subdivide();

/**
 * @brief Insert an agent into the quadtree, subdividing if necessary.
 *
 * Agents near region boundaries are also stored in buffer lists to ensure
 * continuity between neighboring regions. Automatically expands and redistributes
 * agents if the current node exceeds capacity.
 *
 * @param particle The agent data to insert.
 * @param is_particle_placed Internal flag for recursive insertions.
 * @return True if the agent was placed inside this tree.
 */
    bool insert_particle_in_leaf_nodes(const QuadTreeAgentInfo& particle, bool is_inserted);
};

/**
 * @brief Flatten a nested list of values and reorder them according to an index list.
 *
 * This utility function merges several per-leaf vectors into a single contiguous
 * vector and then sorts them based on the provided global agent ID ordering.
 *
 * Commonly used to recombine DRIVE API results after quadtree
 * subdivision.
 *
 * @tparam T Type of elements stored in the nested lists e.g., AgentState, vector<double>.
 * @param nested_list Vector of sublists to flatten.
 * @param index_list index ordering matching agent IDs.
 * @return Sorted vector of elements.
 *
 * @throws std::runtime_error if list sizes do not match.
 */
template <typename T>
std::vector<T> flatten_and_sort(
    const std::vector<std::vector<T>>& nested_list,
    const std::vector<int>& index_list
) {
    // flatten all sublists into one contiguous vector
    std::vector<T> flat_list;
    for (const auto& sublist : nested_list) {
        flat_list.insert(flat_list.end(), sublist.begin(), sublist.end());
    }

    if (flat_list.size() != index_list.size()) {
        throw std::runtime_error("flatten_and_sort error: mismatch between flattened list size ("
                                 + std::to_string(flat_list.size()) + ") and index list size ("
                                 + std::to_string(index_list.size()) + ")");
    }

    std::vector<std::pair<int, T>> zipped;
    zipped.reserve(flat_list.size());
    for (size_t i = 0; i < flat_list.size(); ++i) {
        zipped.emplace_back(index_list[i], flat_list[i]);
    }

    std::sort(zipped.begin(), zipped.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<T> sorted_list;
    sorted_list.reserve(zipped.size());
    for (auto& pair : zipped) {
        sorted_list.push_back(std::move(pair.second));
    }

    return sorted_list;
}

} 
#endif // INVERTEDAI_QUADTREE_H