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
using json = nlohmann::json;
using AttrVariant = std::variant<double, std::string>;

namespace invertedai {

const std::map<std::string, int> kControlType = {
    {"traffic-light", 0},       {"yield", 1},       {"stop-sign", 2},
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
        if (element[2].is_string()) {
            agent_type = element[2];
        }
        else if (element[2].is_number()) {
            rear_axis_offset = element[2];
        }
        else if (element[2].is_array()) {
            waypoint = {element[2][0], element[2][1]};
        }
        else
        {
          throw std::invalid_argument("Invalid data type at position 2.");
        }
        length = element[0];
        width = element[1];
    }
    else if (size == 4)
    {
      length = element[0];
      width = element[1];
      if (element[3].is_array())
      {
        waypoint = {element[3][0], element[3][1]};
        if (element[2].is_string())
        {
          agent_type = element[2];
        }
        else if (element[2].is_number())
        {
          rear_axis_offset = element[2];
        }
        else
        {
          throw std::invalid_argument("Invalid data type at position 2.");
        }
      }
      else if (element[3].is_string()){
      {
        agent_type = element[3];
        if (element[2].is_number())
        {
          rear_axis_offset = element[2];
        }
        else
        {
          rear_axis_offset = 0.0;
        }
      }
    }
    else
        {
          throw std::invalid_argument("Invalid data type at position 3.");
        }
        
    }
    else if(size == 5) {
        length = element[0];
        width = element[1];
        if (element[2].is_number())
        {
          rear_axis_offset = element[2];
        }
        else
        {
          rear_axis_offset = 0.0;
        }
        if (element[3].is_string()) {
            agent_type = element[3];
        }
        waypoint = {element[4][0], element[4][1]};
    }
  }
};

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

} // namespace invertedai

#endif
