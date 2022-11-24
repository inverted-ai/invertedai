load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "client_example",
    srcs = ["client_example.cc"],
    data = [
        "drive_body.json",
        "initialize_body.json",
    ],
    deps = [
        "//invertedai:api",
        "@boost//:beast",
        "@opencv",
    ],
)
