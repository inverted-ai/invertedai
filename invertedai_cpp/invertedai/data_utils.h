#ifndef DATA_UTILS_H
#define DATA_UTILS_H
#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <variant>
#include <iostream>
#include "externals/json.hpp"
#include "error.h"
using json = nlohmann::json;
using AttrVariant = std::variant<double, std::string>;

namespace invertedai {

constexpr int RECURRENT_SIZE = 152;
constexpr double REGION_MAX_SIZE = 100.0;
constexpr double AGENT_SCOPE_FOV_BUFFER = 60.0;
constexpr int ATTEMPT_PER_NUM_REGIONS = 15;

enum class AgentType {
    car,
    pedestrian
    // Add more if needed: bus, truck, cyclist, etc.
};

// Helper to convert to string
inline std::string to_string(AgentType type) {
    switch (type) {
        case invertedai::AgentType::car: return "car";
        case invertedai::AgentType::pedestrian: return "pedestrian";
        default: return "unknown";
    }
}

// Helper to parse from string (if needed)
inline invertedai::AgentType agent_type_from_string(const std::string& s) {
    if (s == "car") return invertedai::AgentType::car;
    if (s == "pedestrian") return invertedai::AgentType::pedestrian;
    throw std::invalid_argument("Unknown AgentType: " + s);
}



const std::map<std::string, int> kControlType = {
    {"traffic_light", 0},       {"yield_sign", 1},       {"stop_sign", 2},
    {"traffic-light-actor", 3}, {"yield-actor", 4}, {"stop-sign-actor", 5}};

/**
 * 2D coordinates of a point in a given location. Each location comes with a
 * canonical coordinate system, where the distance units are meters.
 */
struct Point2d {
  double x, y;
  bool isCloseTo(const Point2d& other, double threshold = 1) const {
        double dx = x - other.x;
        double dy = y - other.y;
        double distance = std::sqrt(dx * dx + dy * dy);
        return distance <= threshold;
    }
};

/**
 * The current or predicted state of a given agent at a given point.
 */
struct AgentState {
  /**
   * The center point of the agent’s bounding box.
   */
  double x, y;
  /**
   * The direction the agent is facing, in radians with 0 pointing along x and
   * pi/2 pointing along y.
   */
  double orientation;
  /**
   * In meters per second, negative if the agent is reversing.
   */
  double speed;
};

/**
 * Static attributes of the agent, which don’t change over the course of a
 * simulation. We assume every agent is a rectangle obeying a kinematic bicycle
 * model.
 */
struct AgentAttributes {
  /**
   * Longitudinal, lateral extent of the agent in meters.
   */
  std::optional<double> length, width;
  /**
   * Distance from the agent’s center to its rear axis in meters. Determines
   * motion constraints.
   */
  std::optional<double> rear_axis_offset;
  /**
   * Agent types are used to indicate how that agent might behave in a scenario.
   * Currently "car" and "pedestrian" are supported.
   */
  std::optional<std::string> agent_type;
  /**
   *  Target waypoint of the agent. If provided the agent will attempt to reach it.
   */
  std::optional<Point2d> waypoint;

  void printFields() const {
    std::cout << "checking fields of current agent..." << std::endl;
    if (length.has_value()) {
      std::cout << "Length: " << length.value() << std::endl;
    }
    if (width.has_value()) {
      std::cout << "Width: " << width.value() << std::endl;
    }
    if (rear_axis_offset.has_value()) {
      std::cout << "rear_axis_offset: " << rear_axis_offset.value() << std::endl;
    }
    if (agent_type.has_value()) {
      std::cout << "Agent type: " << agent_type.value() << std::endl;
    }
    if (waypoint.has_value()) {
      std::cout << "Waypoint: (" << waypoint.value().x << "," << waypoint.value().y << ")"<< std::endl;
    }
  }

  json toJson() const {
    std::vector<AttrVariant> attr_vector;
    if (length.has_value()) {
        attr_vector.push_back(length.value());
    }
    if (width.has_value()) {
        attr_vector.push_back(width.value());
    }
    if (rear_axis_offset.has_value()) {
        attr_vector.push_back(rear_axis_offset.value());
    }
    if (agent_type.has_value()) {
        attr_vector.push_back(agent_type.value());
    }
    std::vector<std::vector<double>> waypoint_vector;
    if (waypoint.has_value()) {  
        waypoint_vector.push_back({waypoint.value().x, waypoint.value().y});
    }
    json jsonArray = json::array();
    for (const auto &element : attr_vector) {
        std::visit([&jsonArray](const auto& value) {
            jsonArray.push_back(value);
        }, element);
    }
    for (const auto &element : waypoint_vector) {
        jsonArray.push_back(element);
    }

    return jsonArray;
  }

  AgentAttributes(const json &element) {
    int size = element.size();
    if (size == 1) {
      if (element[0].is_string()){
        agent_type = element[0];
      }
      else if (element[0].is_array()) {
            waypoint = {element[0][0], element[0][1]};
        }
      else {
        throw std::invalid_argument("Invalid data type at position 0.");
      }
    }
    else if(size == 2) {
        if (element[0].is_string()) {
            agent_type = element[0];
        }
        else {
          throw std::invalid_argument("agent_type must be a string");
        }
        if (element[1].is_array()) {
            waypoint = {element[1][0], element[1][1]};
        }
        else {
          throw std::invalid_argument("Waypoint must be an array of two numbers");
        }
    }
    else if(size == 3) {
        length = element[0];
        width = element[1];
        if (element[2].is_string()) {
            agent_type = element[2];
        }
        else if (element[2].is_number()) {
            rear_axis_offset = element[2];
        }
        else if (element[2].is_array()) {
            waypoint = {element[2][0], element[2][1]};
        }
        else {
          throw std::invalid_argument("Invalid data type at position 2.");
        }
    }
    else if (size == 4) {
      length = element[0];
      width = element[1];
      if (element[3].is_array()) {
        waypoint = {element[3][0], element[3][1]};
        if (element[2].is_string()) {
          agent_type = element[2];
        }
        else if (element[2].is_number()) {
          rear_axis_offset = element[2];
        }
        else {
          throw std::invalid_argument("Invalid data type at position 2.");
        }
      }
      else if (element[3].is_string()) {
        agent_type = element[3];
        if (element[2].is_number()) {
          rear_axis_offset = element[2];
        }
        else {
          rear_axis_offset = 0.0;
        }
      }
      else {
        throw std::invalid_argument("Invalid data type at position 3.");
      }
        
    }
    else if(size == 5) {
        length = element[0];
        width = element[1];
        if (element[2].is_number()) {
          rear_axis_offset = element[2];
        }
        else {
          rear_axis_offset = 0.0;
        }
        if (element[3].is_string()) {
            agent_type = element[3];
        }
        if (element[4].is_array()) {
          waypoint = {element[4][0], element[4][1]};
        }
    }
  }

  AgentAttributes() = default;
};

/**
 * Static agent properties of the agent, which don’t change over the course of a
 * simulation. We assume every agent is a rectangle obeying a kinematic bicycle
 * model.
 * This struct is replacing AgentAttributes.
 */
struct AgentProperties {
    /**
    * Longitudinal, lateral extent of the agent in meters.
    */
    std::optional<double> length, width;
    /**
    * Distance from the agent’s center to its rear axis in meters. Determines
    * motion constraints.
    */
    std::optional<double> rear_axis_offset;
    /**
    * Agent types are used to indicate how that agent might behave in a scenario.
    * Currently "car" and "pedestrian" are supported.
    */
    std::optional<std::string> agent_type;
    /**
    *  Target waypoint of the agent. If provided the agent will attempt to reach it.
    */
    std::optional<Point2d> waypoint;
    /**
    *  Target waypoint of the agent. If provided the agent will attempt to reach it.
    */
    std::optional<double> max_speed;
    /**
    *  Maximum speed limit of the agent in m/s.
    */


    void printFields() const {
        std::cout << "checking fields of current agent..." << std::endl;
        if (length.has_value()) {
          std::cout << "Length: " << length.value() << std::endl;
        }
        if (width.has_value()) {
          std::cout << "Width: " << width.value() << std::endl;
        }
        if (rear_axis_offset.has_value()) {
          std::cout << "rear_axis_offset: " << rear_axis_offset.value() << std::endl;
        }
        if (agent_type.has_value()) {
          std::cout << "Agent type: " << agent_type.value() << std::endl;
        }
        if (waypoint.has_value()) {
          std::cout << "Waypoint: (" << waypoint.value().x << "," << waypoint.value().y << ")"<< std::endl;
        }
        if (max_speed.has_value()) {
          std::cout << "Max speed: " << max_speed.value() << std::endl;
        }
    }

    json toJson() const {
        std::vector<AttrVariant> attr_vector;
        if (length.has_value()) {
            attr_vector.push_back(length.value());
        }
        if (width.has_value()) {
            attr_vector.push_back(width.value());
        }
        if (rear_axis_offset.has_value()) {
            attr_vector.push_back(rear_axis_offset.value());
        }
        if (agent_type.has_value()) {
            attr_vector.push_back(agent_type.value());
        }
        std::vector<std::vector<double>> waypoint_vector;
        if (waypoint.has_value()) {  
            waypoint_vector.push_back({waypoint.value().x, waypoint.value().y});
        }
        json jsonArray = json::array();
        for (const auto &element : attr_vector) {
            std::visit([&jsonArray](const auto& value) {
                jsonArray.push_back(value);
            }, element);
        }
        for (const auto &element : waypoint_vector) {
            jsonArray.push_back(element);
        }
        if (max_speed.has_value()) {
            jsonArray.push_back(max_speed.value());
        }

        return jsonArray;
    }

    AgentProperties(const json &element) {
        if (element.contains(std::string{"length"})){
            if (element["length"].is_number()) {
                length = element["length"];
            } 
        }
        if (element.contains(std::string{"width"})){
            if (element["width"].is_number()) {
                width = element["width"];
            } 
        }
        if (element.contains(std::string{"rear_axis_offset"})){
            if (element["rear_axis_offset"].is_number()) {
                rear_axis_offset = element["rear_axis_offset"];
            } 
        }
        if (element.contains(std::string{"agent_type"})){
            if (element["agent_type"].is_string()) {
                agent_type = element["agent_type"];
            }
        }
        if (element.contains(std::string{"waypoint"})){
            if (element["waypoint"].is_array()) {
                waypoint = {element["waypoint"][0], element["waypoint"][1]};
            }
        }
        if (element.contains(std::string{"max_speed"})){
            if (element["max_speed"].is_number()) {
                max_speed = element["max_speed"];
            }   
        }
    }

    AgentProperties() = default;
};

// Recurrent state used in drive
struct RecurrentState {
  std::vector<float> packed;

  RecurrentState() : packed(RECURRENT_SIZE, 0.0f) {}
  explicit RecurrentState(const std::vector<float>& vals) : packed(vals) {
      if (vals.size() != RECURRENT_SIZE) {
          throw InvertedAIError("RecurrentState must have size 152");
      }
  }
};

struct Agent {
  AgentState state;
  AgentProperties properties;
  RecurrentState recurrent;
};

inline std::string agent_type_to_string(AgentType type) {
  switch (type) {
      case AgentType::car: return "car";
      case AgentType::pedestrian: return "pedestrian";
      // add all your types
      default: return "unknown";
  }
}

// inline std::vector<AgentProperties>
// get_default_agent_properties(const std::map<AgentType,int>& agent_count_dict,
//                              bool use_agent_properties = true) {
//     std::vector<AgentProperties> agent_attributes_list;

//     for (auto& [agent_type, count] : agent_count_dict) {
//         for (int i = 0; i < count; i++) {
//             if (use_agent_properties) {
//                 AgentProperties props;
//                 props.agent_type = agent_type_to_string(agent_type);   //only type
//                 agent_attributes_list.push_back(props);
//             } else {
//                 // if you still support AgentAttributes fallback
//                 AgentProperties props;
//                 props.agent_type = agent_type_to_string(agent_type);  //only type
//                 agent_attributes_list.push_back(props);
//             }
//         }
//     }

//     return agent_attributes_list;
// }


/**
 * Dynamic state of a traffic light.
 */
struct TrafficLightState {
  std::string id;
  std::string value;
};

/**
 * Recurrent state of all the traffic lights in one light group (one intersection).
 */
struct LightRecurrentState {
  float state;
  float time_remaining;
};

/**
 * Infractions committed by a given agent, as returned from invertedai::drive().
 */
struct InfractionIndicator {
  /**
   * True if the agent’s bounding box overlaps with another agent’s bounding
   * box.
   */
  bool collisions;
  /**
   * True if the agent is outside the designated driveable area specified by the
   * map.
   */
  bool offroad;
  /**
   * True if the cross product of the agent’s and its lanelet’s directions is
   * negative.
   */
  bool wrong_way;
};

/**
 * Specifies a traffic light placement. We represent traffic lights as
 * rectangular bounding boxes of the associated stop lines, with orientation
 * matching the direction of traffic going through it.
 */
struct StaticMapActor {
  /**
   * ID as used in invertedai::initialize() and invertedai::drive().
   */
  int actor_id;
  /**
   * Not currently used, there may be more traffic signals in the future.
   */
  std::string agent_type;
  /**
   * The position of the stop line.
   */
  double x, y;
  /**
   * Natural direction of traffic going through the stop line, in radians like
   * in AgentState.
   */
  double orientation;
  /**
   * Size of the stop line, in meters, along its orientation.
   */
  std::optional<double> length, width;
  std::vector<int> dependant;
};

/**
 * Read file from the path, return the content as a string.
 */
std::string read_file(const char *path);

struct Region {
  Point2d center;
  double size;
  double min_x, max_x, min_y, max_y;
  // std::vector<Agent> agents;

  std::vector<AgentState> agent_states;
  std::vector<AgentProperties> agent_properties;
  std::vector<RecurrentState> recurrent_states;

  // Region copy() const {
  //     Region r(center, size);
  //     r.agents = agents; 
  //     return r;
  // }
  Region copy() const {
      Region r(center, size);
      r.agent_states = agent_states;
      r.agent_properties = agent_properties;
      r.recurrent_states = recurrent_states;
      return r;
  }


  static Region create_square_region(
      const Point2d& center,
      double size = 100.0,
      const std::vector< AgentState>& states = {},
      const std::vector< AgentProperties>& props = {},
      const std::vector< RecurrentState>& recurs = {}
  ) {
      if (states.size() != props.size()) {
          throw std::invalid_argument("states and props must have same length");
      }
      if (!recurs.empty() && recurs.size() != states.size()) {
          throw std::invalid_argument("recurs must be empty or same length as states");
      }

      Region region(center, size);
      for (size_t i = 0; i < states.size(); i++) {
          if (!region.is_inside({states[i].x, states[i].y})) {
              throw std::invalid_argument("Agent state outside region.");
          }
          region.agent_states.push_back(states[i]);
          region.agent_properties.push_back(props[i]);
          if (!recurs.empty()) {
              region.recurrent_states.push_back(recurs[i]);
          } else {
              region.recurrent_states.emplace_back(); // default recurrent
          }
      }
      return region;
  }
  Region(
      const Point2d& c, 
      double s
  ) : center(c), size(s) {
      min_x = c.x - s/2;
      max_x = c.x + s/2;
      min_y = c.y - s/2;
      max_y = c.y + s/2;
  }

  //check if point is within an X-Y axis aligned square region
  bool is_inside(const Point2d& p) const {
      return (min_x <= p.x && p.x <= max_x &&
              min_y <= p.y && p.y <= max_y);
  }

//     void insert_agents(Agent&& agent) {
//         if (!is_inside({agent.state.x, agent.state.y})) {
//             throw std::invalid_argument("Agent state outside region");
//         }
//         agents.push_back(std::move(agent));
//     }
//     void clear_agents() {
//         agents.clear();
//     }
  
// };
void insert_agent(
  const  AgentState& state, 
  const  AgentProperties& props, 
  const  RecurrentState& recur = RecurrentState()
) {
  if (!is_inside({state.x, state.y})) {
      throw std::invalid_argument("Agent state outside region");
  }
  agent_states.push_back(state);
  agent_properties.push_back(props);
  recurrent_states.push_back(recur);
}
void clear_agents() {
  agent_states.clear();
  agent_properties.clear();
  recurrent_states.clear();
}

size_t size_agents() const {
  return agent_states.size();
}
};


} // namespace invertedai

#endif
