load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "large_drive_lib",
    srcs = [
        "large_drive.cc",
        "quadtree.cc",
    ],
    hdrs = [
        "large_drive.h",
        "quadtree.h",
    ],
    strip_include_prefix = ".",
    include_prefix = "large/large_drive",
    deps = [
        "//large/large_initialize:large_initialize_lib",
        "//invertedai:error",
        "//invertedai:api",
        "@boost//:beast",
        "@opencv//:opencv",
    ],
    visibility = ["//visibility:public"],
)
