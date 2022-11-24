#include "data_utils.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

std::string read_file(const char *path) {
  std::ifstream f(path);
  std::string str;
  std::ostringstream ss;

  ss << f.rdbuf();
  if (const char *debug_mode = std::getenv("DEBUG")) {
    std::cout << "read_file: " << ss.str() << std::endl;
  }
  return ss.str();
}
