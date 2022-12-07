#include "session.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/error.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/version.hpp>

#include "externals/root_certificates.hpp"

namespace beast = boost::beast; // from <boost/beast.hpp>
namespace http = beast::http;   // from <boost/beast/http.hpp>
namespace net = boost::asio;    // from <boost/asio.hpp>
namespace ssl = net::ssl;       // from <boost/asio/ssl.hpp>
using tcp = net::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

namespace invertedai {

void Session::set_api_key(const std::string &api_key) {
  this->api_key_ = api_key;
}

void Session::connect() {
  if (!SSL_set_tlsext_host_name(this->stream_.native_handle(), this->host_)) {
    beast::error_code ec{static_cast<int>(::ERR_get_error()),
                         net::error::get_ssl_category()};
    throw beast::system_error{ec};
  }
  auto const results = this->resolver_.resolve(this->host_, this->port_);
  beast::get_lowest_layer(this->stream_).connect(results);
  this->stream_.handshake(ssl::stream_base::client);
}

void Session::shutdown() {
  beast::error_code ec;
  this->stream_.shutdown(ec);
  if (ec == net::error::eof) {
    ec = {};
  }
  if (ec) {
    throw beast::system_error{ec};
  }
}

const std::string Session::request(const std::string &mode,
                                   const std::string &body_str,
                                   const std::string &url_query_string) {
  std::string target = "/v0/aws/m1/" + mode + url_query_string;

  http::request<http::string_body> req{
      mode == "location_info" ? http::verb::get : http::verb::post,
      target.c_str(), this->version_};
  req.set(http::field::host, this->host_);
  req.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
  req.set("accept", "application/json");
  req.set("x-api-key", this->api_key_);
  if (debug_mode) {
    std::cout << "req body content:\n";
    std::cout << body_str << std::endl;
  }
  req.body() = body_str;
  req.prepare_payload();

  http::write(this->stream_, req);
  beast::flat_buffer buffer;
  http::response<http::string_body> res;
  beast::error_code ec;
  http::read(this->stream_, buffer, res, ec);
  if (!(res.result() == http::status::ok)) {
    throw std::runtime_error(
        "response status: " + std::to_string(res.result_int()) + "\nbody:\n" +
        res.body());
  }
  if (debug_mode) {
    std::cout << "res body content:\n";
    std::cout << res.body().data() << std::endl;
  }
  return res.body().data();
}

} // namespace invertedai