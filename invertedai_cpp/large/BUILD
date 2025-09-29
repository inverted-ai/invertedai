load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "large_lib",
    srcs = [
        "large_initialize.cc",
        "large_init_helpers.cc",
        # add other .cc files here
    ],
    hdrs = [
        "large_initialize.h",
        "large_init_helpers.h",
        "common.h",
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


cc_binary(
    name = "large_main",
    srcs = ["main.cpp"],
    deps = [
        "//invertedai:error", 
        ":large_lib",
        "//invertedai:api",
        "@boost//:beast",
        "@opencv",
    ],
    # add JSON inputs if needed
    # data = ["drive_body.json", "initialize_body.json"],
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
