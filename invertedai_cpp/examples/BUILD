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
        "@boost//:filesystem",
        "@opencv",
    ],
    copts = ["/std:c++17"],
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
    copts = ["/std:c++17"],
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
    copts = ["/std:c++17"],
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
    copts = ["/std:c++17"],
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
    copts = ["/std:c++17"],
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
    copts = ["/std:c++17"],
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
    copts = ["/std:c++17"],
)