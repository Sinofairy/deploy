/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2021 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/
#ifndef J5_SAMPLE_INCLUDE_MESSAGE_VIO_MESSAGE_H_
#define J5_SAMPLE_INCLUDE_MESSAGE_VIO_MESSAGE_H_
#include <string>
#include <vector>
#include <memory>
#include "hobot/hobot.h"
#include "common/vision_type.h"
#include "hobotlog/hobotlog.hpp"

namespace J5Sample {

#define TYPE_IMAGE_MESSAGE "J5_SAMPLE_IMAGE_MESSAGE"
#define TYPE_DROP_MESSAGE "J5_SAMPLE_DROP_MESSAGE"
#define TYPE_DROP_IMAGE_MESSAGE "J5_SAMPLE_DROP_IMAGE_MESSAGE"
#define TYPE_MULTI_IMAGE_MESSAGE "J5_SAMPLE_MULTI_IMAGE_MESSAGE"
#define TYPE_CODEC_IMAGE_MESSAGE "J5_SAMPLE_CODEC_IAMGE_MESSAGE"

struct VioMessage : public hobot::Message {
 public:
  VioMessage() {}
  virtual ~VioMessage() = default;

  std::string type_ = TYPE_IMAGE_MESSAGE;
  int channel_ = -1;
  // frame id
  uint64_t frame_id_ = 0;
  // time stamp
  uint64_t time_stamp_ = 0;
  // image data
  std::shared_ptr<PyramidFrame> pym_image_;
};

struct MultiVioMessage : VioMessage {
 public:
  MultiVioMessage() {
    LOGI << "MultiVioMessage()";
    type_ = TYPE_MULTI_IMAGE_MESSAGE;}
  std::vector<std::shared_ptr<VioMessage>> multi_vio_img_;
  ~MultiVioMessage() {
    LOGI << "~MultiVioMessage";
    multi_vio_img_.clear();}
};

struct ImageVioMessage : VioMessage {
 public:
  ImageVioMessage() = delete;
  explicit ImageVioMessage(
      const std::shared_ptr<PyramidFrame> &image_frame) {
    type_ = TYPE_IMAGE_MESSAGE;
    if (image_frame) {
      channel_ = image_frame->channel_id_;
      frame_id_ = image_frame->frame_id_;
      time_stamp_ = image_frame->time_stamp_;
      pym_image_ = image_frame;
    } else {
      LOGE << "image frame is nullptr";
    }
  }
  ~ImageVioMessage() {}
};

struct CodecMessage : public hobot::Message {
 public:
  CodecMessage() {}
  virtual ~CodecMessage() = default;

  std::string type_ = TYPE_CODEC_IMAGE_MESSAGE;
  int channel_ = -1;
  // frame id
  uint64_t frame_id_ = 0;
  // time stamp
  uint64_t time_stamp_ = 0;
  // codec image data
  std::shared_ptr<StreamFrame> codec_image_;
};

struct CodecImageMessage : CodecMessage {
 public:
  CodecImageMessage() = delete;
  explicit CodecImageMessage(
      const std::shared_ptr<StreamFrame> &stream_frame) {
    if (stream_frame) {
      channel_ = stream_frame->channel_id_;
      frame_id_ = stream_frame->frame_id_;
      time_stamp_ = stream_frame->time_stamp_;
      codec_image_ = stream_frame;
    } else {
      LOGE << "codec frame is nullptr";
    }
  }
  ~CodecImageMessage() {}
};

}  // namespace J5Sample
#endif  //  J5_SAMPLE_INCLUDE_MESSAGE_VIO_MESSAGE_H_
