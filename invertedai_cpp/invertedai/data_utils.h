#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <map>
#include <optional>
#include <string>
#include <vector>

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
  double length, width;
  /**
   * Distance from the agent’s center to its rear axis in meters. Determines
   * motion constraints.
   */
  double rear_axis_offset;
  /**
   * Agent types are used to indicate how that agent might behave in a scenario.
   * Currently "car" and "pedestrian" are supported.
   */
  std::string agent_type;
};

/**
 * Dynamic state of a traffic light.
 */
struct TrafficLightState {
  std::string id;
  std::string value;
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
