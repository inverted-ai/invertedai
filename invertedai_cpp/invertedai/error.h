#pragma once
#include <exception>
#include <string>
#include <map>
#include <optional>
#include <sstream>

namespace invertedai {

class InvertedAIError : public std::exception {
public:
    InvertedAIError(
        const std::string& message = "<empty message>",
        const std::optional<std::string>& http_body = std::nullopt,
        const std::optional<int>& http_status = std::nullopt,
        const std::optional<std::string>& json_body = std::nullopt,
        const std::map<std::string, std::string>& headers = {},
        const std::optional<std::string>& code = std::nullopt
    );

    const char* what() const noexcept override;
    std::string user_message() const;
    virtual std::string repr() const;

protected:
    std::string message_;
    std::optional<std::string> http_body_;
    std::optional<int> http_status_;
    std::optional<std::string> json_body_;
    std::map<std::string, std::string> headers_;
    std::optional<std::string> code_;
};

// all the subclasses like APIError, APIConnectionError
struct APIError : public InvertedAIError { using InvertedAIError::InvertedAIError; };
struct TryAgain : public InvertedAIError { using InvertedAIError::InvertedAIError; };


struct APIConnectionError : public InvertedAIError {
    APIConnectionError(
        const std::string& message,
        const std::optional<std::string>& http_body = std::nullopt,
        const std::optional<int>& http_status = std::nullopt,
        const std::optional<std::string>& json_body = std::nullopt,
        const std::map<std::string, std::string>& headers = {},
        const std::optional<std::string>& code = std::nullopt,
        bool should_retry = false
    );
    bool should_retry_;
};

struct InvalidRequestError : public InvertedAIError {
    InvalidRequestError(
        const std::string& message,
        const std::string& param,
        const std::optional<std::string>& code = std::nullopt,
        const std::optional<std::string>& http_body = std::nullopt,
        const std::optional<int>& http_status = std::nullopt,
        const std::optional<std::string>& json_body = std::nullopt,
        const std::map<std::string, std::string>& headers = {}
    );
    std::string param;
    std::string repr() const override;
};

struct SignatureVerificationError : public InvertedAIError {
    SignatureVerificationError(
        const std::string& message,
        const std::string& sig_header,
        const std::optional<std::string>& http_body = std::nullopt
    );
    std::string sig_header;
};

} // namespace invertedai
