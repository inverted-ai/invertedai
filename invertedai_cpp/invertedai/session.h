#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <thread>
#include <cstring>

#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/filesystem.hpp>

#include "version.h"
#include "logger.h"

namespace beast = boost::beast; // from <boost/beast.hpp>
namespace net = boost::asio;    // from <boost/asio.hpp>
namespace ssl = net::ssl;       // from <boost/asio/ssl.hpp>
using tcp = net::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

namespace invertedai {

class Session {
private:
  std::string api_key_;
  tcp::resolver resolver_;
  beast::ssl_stream<beast::tcp_stream> ssl_stream_ ;
  beast::tcp_stream tcp_stream_ ;
  const char *debug_mode = std::getenv("DEBUG");
  const char *iai_dev = std::getenv("IAI_DEV");
  const bool local_mode = iai_dev && (std::string(iai_dev) == "1" || std::string(iai_dev) == "True");
  double max_retries = std::numeric_limits<double>::infinity(); // Allows for infinite retries by default
  std::vector<int> status_force_list = {408, 429, 500, 502, 503, 504};
  double base_backoff = 1; // Base backoff time in seconds
  double backoff_factor = 2;
  double current_backoff = base_backoff;
  double max_backoff = 0; // No max backoff by default, 0 signifies no limit
  double jitter_factor = 0.5;

  const char *iai_logger_path_char = std::getenv("IAI_LOGGER_PATH");
  const bool is_log_path_null = iai_logger_path_char == NULL;
  std::string str_path = iai_logger_path_char;
  //Check if path is NULL as well as checking if it ends in a "/"
  const std::filesystem::path iai_logger_path = !is_log_path_null ? (std::strcmp(&str_path.back(),"/") != 0 ? str_path + "/" : str_path) : "./";
  bool is_logging = false;
  
  invertedai::LogWriter logger;

public:
  const char* host_ = local_mode ? "localhost" : "api.inverted.ai";
  const char* port_ = local_mode ? "8000" : "443";
  const char *subdomain = local_mode ? "/" : "/v0/aws/m1/";
  const int version_ = 11;

  explicit Session(net::io_context &ioc, ssl::context &ctx)
      : resolver_(ioc), ssl_stream_(ioc, ctx), tcp_stream_(ioc){
        tcp_stream_.expires_never();
        if (!this->is_log_path_null){
          try {
              if (std::filesystem::create_directory(this->iai_logger_path)) {
                  std::cout << "INFO: Directory created at: " << this->iai_logger_path << std::endl;
              }
              this->is_logging = true;
          } catch (const std::filesystem::filesystem_error& e) {
              std::cout << "WARNING: Could not create a directory for the given path: " << this->iai_logger_path << ". No IAI log will be produced." << std::endl;
          }
          
        }
      };

  ~Session();

  /**
   * Set your own api key here.
   */
  void set_api_key(const std::string &api_key);

  /**
   * Set a specific API URL here.
   */
  void set_url(
    const char* &host,
    const char* &port,
    const char* &subdomain
  );

  /**
   * Connect the session to the host.
   * You can connect once and use the shared session for different request.
   */
  void connect();
  /**
   * Shutdown the session.
   */
  void shutdown();
  /**
   * Use the mode("location_info", "initialize", "drive") to construct a
   * request, sent the request to host, and return the body string of the
   * response.
   */
  const std::string request(
    const std::string &mode,
    const std::string &body_str,
    const std::string &url_params,
    double max_retries = std::numeric_limits<double>::infinity(),
    const std::vector<int>& status_force_list = {408, 429, 500, 502, 503, 504},
    double base_backoff = 1,
    double backoff_factor = 2,
    double max_backoff = 0, // No max by default
    double jitter_factor = 0.5
  );
};

} // namespace invertedai
