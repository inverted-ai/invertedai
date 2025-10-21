load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "large_lib",
    srcs = [
        "large_initialize/large_initialize.cc",
        "large_initialize/large_init_helpers.cc",
    ],
    hdrs = [
        "large_initialize/large_initialize.h",
        "large_initialize/large_init_helpers.h",
    ],
    strip_include_prefix = "large_initialize",
    include_prefix = "large",
    deps = [
        "//invertedai:error",
        "//invertedai:api",
        "@boost//:beast",
        "@opencv",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "large_main",
    srcs = ["main.cpp"],
    deps = [
        ":large_lib",
        "//invertedai:error",
        "//invertedai:api",
        "@boost//:beast",
        "@opencv",
    ],
)

cc_binary(
    name = "large_test_main",
    srcs = ["main.cpp"],
    deps = [
        ":large_lib",
        "//invertedai:api",
        "@boost//:beast",
        "@opencv",
    ],
)
