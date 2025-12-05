load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_library(
    name = "data_utils",
    srcs = ["data_utils.cc"],
    hdrs = ["data_utils.h"],
    deps = [
    "//invertedai/externals:json",
    "//invertedai:error"
    ],
)

cc_library(
    name = "location_info_request",
    srcs = ["location_info_request.cc"],
    hdrs = ["location_info_request.h"],
    deps = [
        ":data_utils",
        "//invertedai/externals:json",
    ],
)

cc_library(
    name = "location_info_response",
    srcs = ["location_info_response.cc"],
    hdrs = ["location_info_response.h"],
    deps = [
        ":data_utils",
        "//invertedai/externals:json",
    ],
)

cc_library(
    name = "initialize_request",
    srcs = ["initialize_request.cc"],
    hdrs = ["initialize_request.h"],
    deps = [
        ":data_utils",
        "//invertedai/externals:json",
    ],
)

cc_library(
    name = "initialize_response",
    srcs = ["initialize_response.cc"],
    hdrs = ["initialize_response.h"],
    deps = [
        ":data_utils",
        "//invertedai/externals:json",
    ],
)

cc_library(
    name = "drive_request",
    srcs = ["drive_request.cc"],
    hdrs = ["drive_request.h"],
    deps = [
        ":data_utils",
        ":drive_response",
        ":initialize_response",
        "//invertedai/externals:json",
    ],
)

cc_library(
    name = "drive_response",
    srcs = ["drive_response.cc"],
    hdrs = ["drive_response.h"],
    deps = [
        ":data_utils",
        "//invertedai/externals:json",
    ],
)

cc_library(
    name = "blame_request",
    srcs = ["blame_request.cc"],
    hdrs = ["blame_request.h"],
    deps = [
        ":data_utils",
        "//invertedai/externals:json",
    ],
)

cc_library(
    name = "blame_response",
    srcs = ["blame_response.cc"],
    hdrs = ["blame_response.h"],
    deps = [
        ":data_utils",
        "//invertedai/externals:json",
    ],
)

cc_library(
    name = "session",
    srcs = ["session.cc"],
    hdrs = ["session.h"],
    linkopts = [
        "-lboost_iostreams",
    ],
    deps = [
        "//invertedai/externals:root-certificates",
        "@boost//:beast",
        ":version",
        ":logger",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "api",
    srcs = ["api.cc"],
    hdrs = ["api.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":blame_request",
        ":blame_response",
        ":drive_request",
        ":drive_response",
        ":initialize_request",
        ":initialize_response",
        ":location_info_request",
        ":location_info_response",
        ":session",
    ],
)

cc_library(
    name = "logger",
    srcs = ["logger.cc"],
    hdrs = ["logger.h"],
    deps = [
        ":drive_request",
        ":drive_response",
        ":initialize_request",
        ":initialize_response",
        ":location_info_request",
        ":location_info_response",
        "//invertedai/externals:json",
    ],
)
cc_library(
    name = "error",
    srcs = ["error.cc"],
    hdrs = ["error.h"],    
    visibility = ["//visibility:public"],
)


cc_library(
    name = "version",
    hdrs = ["version.h"],
    visibility = ["//visibility:public"],
)
