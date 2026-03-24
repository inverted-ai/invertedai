cc_library(
    name = "opencv",
    hdrs = glob([
        "opencv4/opencv2/**/*.h*",
        "x86_64-linux-gnu/opencv4/opencv2/cvconfig.h",
    ]),
    includes = [
        "opencv4",
        "x86_64-linux-gnu/opencv4",
    ],
    linkopts = [
        "-l:libopencv_core.so",
        "-l:libopencv_calib3d.so",
        "-l:libopencv_features2d.so",
        "-l:libopencv_highgui.so",
        "-l:libopencv_imgcodecs.so",
        "-l:libopencv_imgproc.so",
        "-l:libopencv_video.so",
        "-l:libopencv_videoio.so",
    ],
    visibility = ["//visibility:public"],
)
