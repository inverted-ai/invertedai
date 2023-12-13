#!/bin/bash
export IAI_DEV=1

bazel build //examples:drive_tests
bazel build //examples:initialize_tests

./bazel-bin/examples/initialize_tests "EFGHTOKENABCD"
./bazel-bin/examples/drive_tests 30 "EFGHTOKENABCD"
