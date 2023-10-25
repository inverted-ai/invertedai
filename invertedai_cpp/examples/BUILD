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