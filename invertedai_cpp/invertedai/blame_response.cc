#include "blame_response.h"

using json = nlohmann::json;

namespace invertedai {

BlameResponse::BlameResponse(const std::string &body_str) {
  this->body_json_ = json::parse(body_str);

  this->agents_at_fault_.clear();
  for (const auto &element : this->body_json_["agents_at_fault"]) {
    this->agents_at_fault_.push_back(element.get<int>());
  }
  this->confidence_score_ = this->body_json_["confidence_score"].is_number()
    ? std::optional<float>{this->body_json_["confidence_score"].get<float>()}
    : std::nullopt;
  if (this->body_json_["reasons"].is_null()) {
    this->reasons_ = std::nullopt;
  } else {
    std::map<int, std::vector<std::string>> reasons;
    reasons.clear();
    for (const auto &pair : this->body_json_["reasons"].get<std::map<std::string, std::vector<std::string>>>()) {
      reasons.insert(std::make_pair(std::stoi(pair.first), pair.second));
    }
    this->reasons_ = std::optional<std::map<int, std::vector<std::string>>>{reasons};
  }
  if (this->body_json_["birdviews"].is_null()) {
    this->birdviews_ = std::nullopt;
  } else {
    this->birdviews_ = std::optional<std::vector<std::vector<unsigned char>>>{
      this->body_json_["birdviews"].get<std::vector<std::vector<unsigned char>>>()
    };
  }
}

void BlameResponse::refresh_body_json_() {
  this->body_json_["agents_at_fault"] = this->agents_at_fault_;
  if (this->confidence_score_.has_value()) {
    this->body_json_["confidence_score"] = this->confidence_score_.value();
  } else {
    this->body_json_["confidence_score"] = nullptr;
  }
  if (this->reasons_.has_value()) {
    this->body_json_["reasons"] = json::object();
    for (const auto &pair : this->reasons_.value()) {
      this->body_json_["reasons"][std::to_string(pair.first)] = pair.second;
    }
  } else {
    this->body_json_["reasons"] = nullptr;
  }
  if (this->birdviews_.has_value()) {
    this->body_json_["birdviews"] = this->birdviews_.value();
  } else {
    this->body_json_["birdviews"] = nullptr;
  }
}

std::string BlameResponse::body_str() {
  this->refresh_body_json_();
  return this->body_json_.dump();
}

std::vector<int> BlameResponse::agents_at_fault() const {
  return this->agents_at_fault_;
}

std::optional<float> BlameResponse::confidence_score() const {
  return this->confidence_score_;
}

std::optional<std::map<int, std::vector<std::string>>> BlameResponse::reasons() const {
  return this->reasons_;
}

std::optional<std::vector<std::vector<unsigned char>>> BlameResponse::birdviews() const {
  return this->birdviews_;
}

void BlameResponse::set_agents_at_fault(const std::vector<int> &agents_at_fault) {
  this->agents_at_fault_ = agents_at_fault;
}

void BlameResponse::set_confidence_score(std::optional<float> confidence_score) {
  this->confidence_score_ = confidence_score;
}

void BlameResponse::set_reasons(const std::optional<std::map<int, std::vector<std::string>>> &reasons) {
  this->reasons_ = reasons;
}

void BlameResponse::set_birdviews(const std::optional<std::vector<std::vector<unsigned char>>> &birdviews) {
  this->birdviews_ = birdviews;
}

} // namespace invertedai
