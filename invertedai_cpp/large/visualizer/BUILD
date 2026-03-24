load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "visualizer_helpers",
    hdrs = ["visualizer_helpers.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "visualizer_lib",
    srcs = ["visualizer.cc"],
    hdrs = ["visualizer.h"],
    deps = [
        ":visualizer_helpers",
        "//invertedai:api",
        "//invertedai:session",
        "//invertedai:error",
        "//large/large_drive:large_drive_lib",
        "//large/large_initialize:large_initialize_lib",
        "@opencv//:opencv",
    ],
    visibility = ["//visibility:public"],
)
