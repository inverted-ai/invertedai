#include <string>
#include <vector>

#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>
#include "version.h"

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

public:
  const char* host_ = local_mode ? "localhost" : "api.inverted.ai";
  const char* port_ = local_mode ? "8000" : "443";
  const char *subdomain = local_mode ? "/" : "/v0/aws/m1/";;
  const int version_ = 11;

  explicit Session(net::io_context &ioc, ssl::context &ctx)
      : resolver_(ioc), ssl_stream_(ioc, ctx), tcp_stream_(ioc){};

  /**
   * Set your own api key here.
   */
  void set_api_key(const std::string &api_key);

  /**
   * Set a specific API URL here.
   */
  void set_url(
    const std::string &host,
    const std::string &port,
    const std::string &subdomain
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
    const std::string &url_params
  );
};

} // namespace invertedai
