load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "large_initialize_lib",
    srcs = [
        "large_initialize.cc",
        "large_init_helpers.cc",
    ],
    hdrs = [
        "large_initialize.h",
        "large_init_helpers.h",
    ],
    strip_include_prefix = ".",
    include_prefix = "large/large_initialize",
    deps = [
        "//invertedai:error",
        "//invertedai:api",
        "@boost//:beast",
        "@opencv//:opencv",
    ],
    visibility = ["//visibility:public"],
)
