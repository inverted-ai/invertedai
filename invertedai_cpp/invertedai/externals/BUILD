load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_library(
    name = "root-certificates",
    visibility = ["//visibility:public"],
    hdrs = ["root_certificates.hpp"],
    deps = ["@boost//:asio_ssl"]
)

cc_library(
    name = "json",
    visibility = ["//visibility:public"],
    hdrs = ["json.hpp"]
)
