#include <string>
#include <vector>

#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>

namespace beast = boost::beast; // from <boost/beast.hpp>
namespace net = boost::asio;    // from <boost/asio.hpp>
namespace ssl = net::ssl;       // from <boost/asio/ssl.hpp>
using tcp = net::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

class Session {
private:
  const char *host_ = "api.inverted.ai";
  const char *port_ = "443";
  const int version_ = 11;
  std::string api_key_;
  tcp::resolver resolver_;
  beast::ssl_stream<beast::tcp_stream> stream_;
  const char *debug_mode = std::getenv("DEBUG");

public:
  explicit Session(net::io_context &ioc, ssl::context &ctx)
      : resolver_(ioc), stream_(ioc, ctx){};

  void set_api_key(const std::string &api_key);
  void connect();
  void shutdown();
  const std::string request(const std::string &mode,
                            const std::string &body_str,
                            const std::string &url_params);
};
