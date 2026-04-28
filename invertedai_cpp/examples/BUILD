load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "client_example",
    srcs = ["client_example.cc"],
    data = [
        "drive_body.json",
        "initialize_body.json",
        "location_info_body.json",
    ],
    deps = [
        "//invertedai:api",
        "@boost//:beast",
        "@opencv",
    ],
)

cc_binary(
    name = "blame_example",
    srcs = ["blame_example.cc"],
    data = [
        "drive_body.json",
        "initialize_body.json",
        "location_info_body.json",
        "blame_body.json",
    ],
    deps = [
        "//invertedai:api",
        "@boost//:beast",
        "@opencv",
    ],
)

cc_binary(
    name = "fps_control_demo",
    srcs = ["fps_control_demo.cc"],
    data = [
        "drive_body.json",
        "initialize_body.json",
        "location_info_body.json",
    ],
    deps = [
        "//invertedai:api",
        "@boost//:beast",
        "@opencv",
    ],
)

cc_binary(
    name = "initialize_tests",
    srcs = ["initialize_tests.cc"],
    data = [
        "initialize_body.json",
        "initialize_sampling_with_types.json",
        "initialize_with_states_and_attributes.json",
    ],
    deps = [
        "//invertedai:api",
        "@boost//:beast",
        "@opencv",
    ],
)

cc_binary(
    name = "drive_tests",
    srcs = ["drive_tests.cc"],
    data = [
        "drive_body.json",
        "initialize_body.json",
        "initialize_sampling_with_types.json",
        "initialize_with_states_and_attributes.json",
    ],
    deps = [
        "//invertedai:api",
        "@boost//:beast",
        "@opencv",
    ],
)

cc_binary(
    name = "waypoint_example",
    srcs = ["waypoint_example.cc"],
    data = [
        "drive_body.json",
        "initialize_body_waypoint_car_example.json",
    ],
    deps = [
        "//invertedai:api",
        "@boost//:beast",
        "@opencv",
    ],
)

cc_binary(
    name = "max_speed_example",
    srcs = ["max_speed_example.cc"],
    data = [
        "drive_body.json",
        "initialize_body_max_speed_car_example.json",
    ],
    deps = [
        "//invertedai:api",
        "@boost//:beast",
        "@opencv",
    ],
)

cc_binary(
    name = "large_example",
    srcs = ["large_example.cc"],
    deps = [
        "//large/visualizer:visualizer_lib",
        "//large/large_drive:large_drive_lib",
        "//large/large_initialize:large_initialize_lib",
        "//invertedai:api",
        "//invertedai:session",
        "//invertedai:error",
        "@boost//:asio",
        "@opencv//:opencv",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "scenario_log_example",
    srcs = ["scenario_log_example.cc"],
    deps = [
        "//invertedai:api",
        "//invertedai:session",
        "//invertedai:logger",
        "//invertedai:location_info_request",
        "//invertedai:location_info_response",
        "//invertedai:drive_request",
        "//invertedai:drive_response",
        "//invertedai:visualize",
        "@boost//:asio",
        "@opencv//:opencv",
        "//invertedai/externals:json",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "scenario_replay_ego_takeover_example",
    srcs = ["scenario_replay_ego_takeover_example.cc"],
    data = [
        "carla_Town10HD_log.json",
    ],
    deps = [
        "//invertedai:api",
        "//invertedai:session",
        "//invertedai:logger",
        "//invertedai:visualize",
        "//invertedai:location_info_request",
        "//invertedai:location_info_response",
        "//invertedai:drive_request",
        "//invertedai:drive_response",
        "//invertedai:initialize_request",
        "//invertedai:initialize_response",
        "//large/large_drive:large_drive_lib",
        "//large/large_initialize:large_initialize_lib",
        "//invertedai/externals:json",
        "@boost//:asio",
        "@opencv//:opencv",
    ],
    visibility = ["//visibility:public"],
)

