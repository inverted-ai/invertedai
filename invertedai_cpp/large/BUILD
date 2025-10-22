load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "large_initialize_lib",
    srcs = [
        "large_initialize/large_initialize.cc",
        "large_initialize/large_init_helpers.cc",
    ],
    hdrs = [
        "large_initialize/large_initialize.h",
        "large_initialize/large_init_helpers.h",
    ],
    strip_include_prefix = "large_initialize",
    include_prefix = "large/large_initialize",
    deps = [
        "//invertedai:error",
        "//invertedai:api",
        "@boost//:beast",
        "@opencv",
    ],
    visibility = ["//visibility:public"],
)


cc_library(
    name = "large_drive_lib",
    srcs = [
        "large_drive/large_drive.cc",
        "large_drive/quadtree.cc",
    ],
    hdrs = [
        "large_drive/large_drive.h",
        "large_drive/quadtree.h",
    ],
    strip_include_prefix = "large_drive",
    include_prefix = "large/large_drive",
    deps = [
        ":large_initialize_lib",  # depends on initialization logic
        "//invertedai:error",
        "//invertedai:api",
        "@boost//:beast",
        "@opencv",
    ],
    visibility = ["//visibility:public"],
)


cc_binary(
    name = "large_main",
    srcs = ["visualizer.cc"],
    deps = [
        ":large_drive_lib",     # build (drive + initialize)
        ":large_initialize_lib",
        "//invertedai:error",
        "//invertedai:api",
        "//invertedai:session",
        "@boost//:asio",
        "@opencv//:opencv",
    ],
    visibility = ["//visibility:public"],
)
