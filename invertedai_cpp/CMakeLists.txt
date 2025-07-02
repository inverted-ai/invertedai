cmake_minimum_required(VERSION 3.15)
project(invertedai_cpp)

set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/../invertedai_cpp
    ${OpenCV_INCLUDE_DIRS}
)

# Add Boost, OpenSSL, etc. if needed
find_package(OpenSSL REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem iostreams)
find_package(OpenCV REQUIRED)


# Add executable
add_executable(
    client_example 
    examples/client_example.cpp
    invertedai/api.cpp
    invertedai/blame_request.cpp
    invertedai/blame_response.cpp
    invertedai/data_utils.cpp
    invertedai/drive_request.cpp
    invertedai/drive_response.cpp
    invertedai/initialize_request.cpp
    invertedai/initialize_response.cpp
    invertedai/location_info_request.cpp
    invertedai/location_info_response.cpp
    invertedai/logger.cpp
    invertedai/session.cpp
)

# Link libraries
target_link_libraries(client_example PRIVATE
    OpenSSL::SSL
    OpenSSL::Crypto
    Boost::system
    Boost::filesystem
    Boost::iostreams
    ${OpenCV_LIBS}
)
