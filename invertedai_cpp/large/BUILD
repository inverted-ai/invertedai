load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "large_lib",
    srcs = [
        "large_initialize.cc",
        "large_init_helpers.cc",
        "large_drive.cc",
        "quadtree.cc",
        # add other .cc files here
    ],
    hdrs = [
        "large_initialize.h",
        "large_init_helpers.h",
        "large_drive.h",
        "common.h",
        "quadtree.h",
        # add other headers you want available
    ],
    strip_include_prefix = ".",
    include_prefix = "large",   # so #include "large/large_initialize.h" works
    deps = [
        "//invertedai:error", 
        "//invertedai:api",
        "@boost//:beast",
        "@opencv",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "large_common",
    hdrs = ["common.h"],
    includes = ["."],   # allows #include "large/common.h"
    visibility = ["//visibility:public"],
)

cc_library(
    name = "large_drive",
    srcs = ["large_drive.cc"],
    hdrs = ["large_drive.h"],
    deps = [
        "//invertedai:api",
        "//invertedai:session",
        "//invertedai:error",
        ":quadtree",
        ":large_lib",
        "//large:large_common",
        "@opencv//:opencv",
        "@boost//:asio",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "large_main",
    srcs = ["main.cpp"],
    deps = [
        ":large_lib",
        "//invertedai:api",
        "//invertedai:session",
        "@boost//:asio",
        "@opencv//:opencv",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "quadtree",
    srcs = ["quadtree.cc"],
    hdrs = ["quadtree.h"],
    deps = [
        "//invertedai:api",
        "//large:large_common",
    ],
    visibility = ["//visibility:public"],
)
cc_binary(
    name = "large_test_main",
    srcs = ["main.cpp"],
    deps = [
        "//large:large_common",
        "//invertedai:api",
        "@boost//:beast",
        "@opencv",
    ],
)

cc_binary(
    name = "test_large_initialize_draw",
    srcs = ["test_large_initialize_draw.cc"],
    deps = [
        ":large_lib",             # your main large_initialize target
        "//large:large_common",
        "//invertedai:api",
        "@opencv//:opencv",        # adjust if you vendored OpenCV differently
        "@boost//:asio",           # same for boost-asio
    ],
)
