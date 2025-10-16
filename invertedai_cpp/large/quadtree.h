#ifndef INVERTEDAI_QUADTREE_H
#define INVERTEDAI_QUADTREE_H

#include <vector>
#include <optional>
#include "common.h"
#include "invertedai/data_utils.h"    
namespace invertedai {

constexpr double BUFFER_FOV = 35.0;
constexpr double QUADTREE_SIZE_BUFFER = 1.0;

struct QuadTreeAgentInfo {
    AgentState agent_state;
    std::optional<std::vector<double>> recurrent_state;
    AgentProperties agent_properties;
    int agent_id;

    std::vector<std::variant<AgentState, AgentProperties, std::optional<std::vector<double>>, int>> 
    tolist() const {
        return {agent_state, agent_properties, recurrent_state, agent_id};
    }

    static QuadTreeAgentInfo fromlist(
        const std::vector<std::variant<AgentState, AgentProperties, std::optional<std::vector<double>>, int>>& l
    ) {
        if (l.size() != 4) {
            throw std::invalid_argument("fromlist requires exactly 4 elements");
        }
        QuadTreeAgentInfo q;
        q.agent_state      = std::get<AgentState>(l[0]);
        q.agent_properties = std::get<AgentProperties>(l[1]);
        q.recurrent_state  = std::get<std::optional<std::vector<double>>>(l[2]);
        q.agent_id         = std::get<int>(l[3]);
        return q;
    }
};

class QuadTree {
public:
    QuadTree(int capacity, const Region& region);

    bool insert(const QuadTreeAgentInfo& particle, bool is_particle_placed = false);
    std::vector<invertedai::Region> get_regions() const;
    std::vector<QuadTree*> get_leaf_nodes();
    size_t get_number_of_agents_in_node() const;
    const std::vector<QuadTreeAgentInfo>& particles() const { return particles_; }
    const std::vector<QuadTreeAgentInfo>& particles_buffer() const { return particles_buffer_; }
    const Region& region() const { return region_; }
    const Region& region_buffer() const { return region_buffer_; }

private:
    int capacity_;
    invertedai::Region region_;
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

/// utility function
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