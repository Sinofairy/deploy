/**
 * Copyright Horizon Robotics
 */

#ifndef J5_SAMPLE_INCLUDE_MESSAGE_BASE_MESSAGE_H__
#define J5_SAMPLE_INCLUDE_MESSAGE_BASE_MESSAGE_H__

#include <stdint.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "hobot/hobot.h"
#include "message/vio_message.h"
#include "common/profile.h"

namespace J5Sample {

typedef std::shared_ptr<ImageFrame> spImageFrame;
typedef std::shared_ptr<PyramidFrame> spPyramidFrame;
typedef std::shared_ptr<StreamFrame> spStreamFrame;
typedef std::shared_ptr<VioMessage> spVioMessage;
typedef std::shared_ptr<CodecImageMessage> spCodecImageMessage;

enum struct DataState {
  /// valid
  VALID = 0,
  /// filtered
  FILTERED = 1,
  /// invisible
  INVISIBLE = 2,
  /// disappeared
  DISAPPEARED = 3,
  /// invalid
  INVALID = 4,
};

template <typename Dtype>
struct BBox_ : public hobot::Message {
  Dtype x1_;
  Dtype y1_;
  Dtype x2_;
  Dtype y2_;
  float score_ = 0.0;
  int32_t id_ = 0;
  float rotation_angle_ = 0.0;
  std::string specific_type_ = "";
  std::string type_ = "";

  inline BBox_() { type_ = "BBox"; }
  inline BBox_(Dtype x1, Dtype y1, Dtype x2, Dtype y2, float score = 0.0f,
               int32_t id = -1, const std::string &specific_type = "") {
    x1_ = x1;
    y1_ = y1;
    x2_ = x2;
    y2_ = y2;
    id_ = id;
    score_ = score;
    specific_type_ = specific_type;
    type_ = "BBox";
  }
  inline Dtype Width() const { return (x2_ - x1_); }
  inline Dtype Height() const { return (y2_ - y1_); }
  inline Dtype CenterX() const { return (x1_ + (x2_ - x1_) / 2); }
  inline Dtype CenterY() const { return (y1_ + (y2_ - y1_) / 2); }

  inline friend std::ostream &operator<<(std::ostream &out, BBox_ &bbox) {
    out << "( x1: " << bbox.x1_ << " y1: " << bbox.y1_ << " x2: " << bbox.x2_
        << " y2: " << bbox.y2_ << " score: " << bbox.score_ << " )";
    return out;
  }
};
typedef BBox_<float> BBox;
typedef std::shared_ptr<BBox> spBBox;

template <typename DType>
struct Attribute_ : public hobot::Message {
  DType value_;
  float score_ = 0.0;
  std::string specific_type_ = "";
  std::string type_ = "";

  Attribute_() { type_ = "Attribute"; }
  Attribute_(int value, float score, const std::string specific_type = "")
      : value_(value), score_(score), specific_type_(specific_type) {
    type_ = "Attribute";
  }
  friend bool operator>(const Attribute_ &lhs, const Attribute_ &rhs) {
    return (lhs.score_ > rhs.score_);
  }
  inline friend std::ostream &operator<<(std::ostream &out, Attribute_ &attr) {
    out << "(value: " << attr.value_ << ", score: " << attr.score_;
    return out;
  }
};
typedef Attribute_<int32_t> Gender;          // 性别
typedef Attribute_<int32_t> Age;             // 性别
typedef Attribute_<int32_t> Classification;  // 分类
typedef std::shared_ptr<Classification> spClassification;

/**
 * \~Chinese @brief 人体分割
 */
struct Segmentation : public hobot::Message {
  std::vector<int> values_;
  std::vector<float> pixel_score_;
  std::string specific_type_ = "";
  int32_t width_ = 0;
  int32_t height_ = 0;
  float score_ = 0.0;
  std::string type_ = "";

  Segmentation() { type_ = "Segmentation"; }
  inline friend std::ostream &operator<<(std::ostream &out, Segmentation &seg) {
    out << "(";
    for (auto value : seg.values_) out << value;
    out << ")";
    return out;
  }
};
typedef std::shared_ptr<Segmentation> spSegmentation;

struct Target {
  std::string type_;
  std::vector<spBBox> boxs_;
  std::vector<spClassification> attrs_;
  std::vector<spSegmentation> segs_;
};
typedef std::shared_ptr<Target> spTarget;
struct J5FrameMessage : public hobot::Message {
  J5FrameMessage() {}
  ~J5FrameMessage() {}

  uint32_t channel_id_ = 0;
  uint64_t time_stamp_ = 0;
  uint64_t frame_id_ = 0;
  uint64_t sequence_id_ = 0;
  spVioMessage pym_img_;
  std::vector<spTarget> targets_;
  //  profile
  spModuleProfile web_display_profile_ =  nullptr;
};
typedef std::shared_ptr<J5FrameMessage> spJ5FrameMessage;
}  // namespace J5Sample
#endif  //  J5_SAMPLE_INCLUDE_MESSAGE_BASE_MESSAGE_H__

