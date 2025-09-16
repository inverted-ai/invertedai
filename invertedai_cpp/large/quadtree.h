#ifndef INVERTEDAI_QUADTREE_H
#define INVERTEDAI_QUADTREE_H

#include <vector>
#include <optional>
#include "externals/json.hpp"
#include "common.h"

namespace invertedai {

constexpr double BUFFER_FOV = 35.0;
constexpr double QUADTREE_SIZE_BUFFER = 1.0;

struct QuadTreeAgentInfo {
    AgentState agent_state;
    AgentProperties agent_properties;
    std::optional<RecurrentState> recurrent_state;
    int agent_id;

    std::vector<std::variant<AgentState, AgentProperties, std::optional<RecurrentState>, int>> tolist() const;
    static QuadTreeAgentInfo fromlist(const AgentState& state,
                                      const AgentProperties& props,
                                      const std::optional<RecurrentState>& recur,
                                      int id);
};

class QuadTree {
public:
    QuadTree(int capacity, const Region& region);

    bool insert(const QuadTreeAgentInfo& particle, bool is_particle_placed = false);
    std::vector<Region> get_regions() const;
    std::vector<QuadTree*> get_leaf_nodes();
    size_t get_number_of_agents_in_node() const;

private:
    int capacity_;
    Region region_;
    Region region_buffer_;
    bool leaf_;

    std::vector<QuadTreeAgentInfo> particles_;
    std::vector<QuadTreeAgentInfo> particles_buffer_;
    std::unique_ptr<QuadTree> northWest_;
    std::unique_ptr<QuadTree> northEast_;
    std::unique_ptr<QuadTree> southWest_;
    std::unique_ptr<QuadTree> southEast_;

    void subdivide();
    bool insert_particle_in_leaf_nodes(const QuadTreeAgentInfo& particle, bool is_inserted);
};

/// Utility function
template <typename T>
std::vector<T> flatten_and_sort(const std::vector<std::vector<T>>& nested_list,
                                const std::vector<int>& index_list);

} // namespace invertedai
