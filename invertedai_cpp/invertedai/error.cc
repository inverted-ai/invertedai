#include "error.h"
    


namespace invertedai {

InvertedAIError::InvertedAIError(
    const std::string& message,
    const std::optional<std::string>& http_body,
    const std::optional<int>& http_status,
    const std::optional<std::string>& json_body,
    const std::map<std::string, std::string>& headers,
    const std::optional<std::string>& code
)
: message_(message),
  http_body_(http_body),
  http_status_(http_status),
  json_body_(json_body),
  headers_(headers),
  code_(code) {}

const char* InvertedAIError::what() const noexcept {
    return message_.c_str();
}

std::string InvertedAIError::user_message() const {
    return message_;
}

std::string InvertedAIError::repr() const {
    std::ostringstream oss;
    oss << typeid(*this).name() 
        << "(message=" << message_
        << ", http_status=" << (http_status_.has_value() ? std::to_string(http_status_.value()) : "null")
        << ")";
    return oss.str();
}

APIConnectionError::APIConnectionError(
    const std::string& message,
    const std::optional<std::string>& http_body,
    const std::optional<int>& http_status,
    const std::optional<std::string>& json_body,
    const std::map<std::string, std::string>& headers,
    const std::optional<std::string>& code,
    bool should_retry
)
: InvertedAIError(message, http_body, http_status, json_body, headers, code),
  should_retry_(should_retry) {}

InvalidRequestError::InvalidRequestError(
    const std::string& message,
    const std::string& param,
    const std::optional<std::string>& code,
    const std::optional<std::string>& http_body,
    const std::optional<int>& http_status,
    const std::optional<std::string>& json_body,
    const std::map<std::string, std::string>& headers
)
: InvertedAIError(message, http_body, http_status, json_body, headers, code),
  param(param) {}

std::string InvalidRequestError::repr() const {
    std::ostringstream oss;
    oss << "InvalidRequestError(message=" << message_
        << ", param=" << param
        << ", code=" << (code_.has_value() ? code_.value() : "null")
        << ", http_status=" << (http_status_.has_value() ? std::to_string(http_status_.value()) : "null")
        << ")";
    return oss.str();
}

SignatureVerificationError::SignatureVerificationError(
    const std::string& message,
    const std::string& sig_header,
    const std::optional<std::string>& http_body
)
: InvertedAIError(message, http_body),
  sig_header(sig_header) {}

} // namespace invertedai
