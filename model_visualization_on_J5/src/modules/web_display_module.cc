/**
 * Copyright Horizon Robotics
 */
#include "web_display_module.h"

#include "hobotlog/hobotlog.hpp"
#include "protocol/x3.pb.h"

namespace J5Sample {

int WebDisplayModule::Init(hobot::RunContext *context) {
  data_send_thread_.CreatThread(1);
  uws_server_ = std::make_shared<UwsServer>();
  if (uws_server_->Init()) {
    LOGE << "UwsPlugin Init uWS server failed";
    return -1;
  }
  worker_ = std::make_shared<std::thread>(
      std::bind(&WebDisplayModule::map_smart_proc, this));
  profile_ = std::make_shared<ModuleProfile>("fps");
  LOGD << "WebDisplayModule Init Suc.";
  return 0;
}

void WebDisplayModule::Reset() {
  if (worker_ && worker_->joinable()) {
    map_stop_ = true;
    map_smart_condition_.notify_one();
    worker_->join();
    worker_ = nullptr;
    LOGI << "WebDisplayModule stop worker";
  }
  if (uws_server_) {
    uws_server_->DeInit();
    uws_server_ = nullptr;
  }
}

FORWARD_DEFINE(WebDisplayModule, 0) {
  spCodecImageMessage codec_img = nullptr;
  spJ5FrameMessage smart_msg = nullptr;
  if (!(*input[0]).empty()) {
    smart_msg = std::dynamic_pointer_cast<J5FrameMessage>((*input[0])[0]);
  }
  if (!(*input[1]).empty()) {
    codec_img = std::dynamic_pointer_cast<CodecImageMessage>((*input[1])[0]);
  }
  LOGD << "codec_img = " << codec_img << ", smart_msg = " << smart_msg;
  {
    {
      std::lock_guard<std::mutex> smart_lock(map_smart_mutex_);
      if (codec_img) {
        spCodecData codec_data(new CodecData(), [&](CodecData *p) {
          if (p) {
            if (p->stream_vaddr_ != 0) {
              LOGD << "webdisplay delete codec image frame";
              std::free(reinterpret_cast<uint8_t *>(p->stream_vaddr_));
            }
            delete (p);
            p = nullptr;
          }
        });
        codec_data->channel_ = codec_img->channel_;
        codec_data->time_stamp_ = codec_img->time_stamp_;
        codec_data->frame_id_ = codec_img->frame_id_;
        codec_data->src_width_ =
            codec_img->codec_image_->stream_info_.src_width;
        codec_data->src_height_ =
            codec_img->codec_image_->stream_info_.src_height;
        codec_data->stream_size_ =
            codec_img->codec_image_->stream_info_.stream_size;
        codec_data->stream_pts_ =
            codec_img->codec_image_->stream_info_.stream_pts;
        uint8_t *buf = reinterpret_cast<uint8_t *>(
            std::calloc(1, codec_data->stream_size_));
        memcpy(buf,
               reinterpret_cast<const char *>(
                   codec_img->codec_image_->stream_info_.stream_vaddr),
               codec_data->stream_size_);
        codec_data->stream_vaddr_ = reinterpret_cast<uint64_t>(buf);
        j5_frames_.push(codec_data);
      }
      if (smart_msg) {
        profile_->FrameStatistic();
        smart_msg->web_display_profile_ =
            std::make_shared<ModuleProfile>("WebDisplayModule");
        j5_smart_msg_.push(smart_msg);
      }
    }
    map_smart_condition_.notify_one();
  }
  // do something
  LOGI << "WebDisplayModule Forward return\n";
  workflow->Return(this, 0, nullptr, context);
}

std::string WebDisplayModule::Serilize(spJ5FrameMessage msg,
                                       spCodecData codec_frame) {
  std::string proto_str;
  x3::FrameMessage proto_frame_message;
  proto_frame_message.set_timestamp_(msg->time_stamp_);
  // img
  if (codec_frame && codec_frame->stream_vaddr_ &&
      codec_frame->stream_size_ > 0) {
    int len = codec_frame->stream_size_;
    LOGD << "codec image len = " << len;
    auto image = proto_frame_message.mutable_img_();
    auto image_str = image->mutable_buf_();
    image_str->resize(len);
    memcpy(const_cast<char *>(image_str->data()),
           reinterpret_cast<const char *>(codec_frame->stream_vaddr_), len);
    image->set_type_("jpg");
    image->set_width_(codec_frame->src_width_);
    image->set_height_(codec_frame->src_height_);
  } else {
    LOGE << "codec img data invalid.";
  }
  // smart_msg
  auto smart_msg = proto_frame_message.mutable_smart_msg_();
  smart_msg->set_timestamp_(msg->time_stamp_);
  smart_msg->set_error_code_(0);
  smart_msg->set_sequence_id_(msg->sequence_id_);
  smart_msg->set_channel_id_(msg->channel_id_);
  smart_msg->set_frame_id_(msg->frame_id_);
  for (auto org_target : msg->targets_) {
    auto dst_target = smart_msg->add_targets_();
    dst_target->set_type_(org_target->type_);
    /* box */
    for (auto org_box : org_target->boxs_) {
      LOGD << *org_box;
      auto dst_box = dst_target->add_boxes_();
      dst_box->set_type_(org_box->type_);
      dst_box->set_score_(org_box->score_);
      dst_box->set_name_("BBox");
      dst_box->set_specific_type_(org_box->specific_type_);
      auto point1 = dst_box->mutable_top_left_();
      point1->set_x_(org_box->x1_);
      point1->set_y_(org_box->y1_);
      auto point2 = dst_box->mutable_bottom_right_();
      point2->set_x_(org_box->x2_);
      point2->set_y_(org_box->y2_);
    }
    for (auto org_seg : org_target->segs_) {
      if (org_seg->width_ * org_seg->height_ ==
          static_cast<int>(org_seg->values_.size())) {
        auto dst_seg = dst_target->add_float_matrixs_();
        dst_seg->set_type_("segmentation");
        dst_seg->set_score_(org_seg->score_);
        dst_seg->set_name_("Segmentation");
        dst_seg->set_specific_type_(org_seg->specific_type_);
        for (int h = 0; h < org_seg->height_; h++) {
          auto float_array = dst_seg->add_arrays_();
          auto base_index = h * org_seg->width_;
          for (int w = 0; w < org_seg->width_; w++) {
            float_array->add_value_(org_seg->values_[base_index + w]);
          }
        }
      } else {
        LOGE << "Seg value has errors!";
      }
    }
  }
  proto_frame_message.SerializeToString(&proto_str);
  return std::move(proto_str);
}

int WebDisplayModule::SendSmartMessage(spJ5FrameMessage msg,
                                       spCodecData codec_frame) {
  auto msg_data = Serilize(msg, codec_frame);
  if (msg_data.empty()) {
    LOGE << "Serilize Error.";
    return -1;
  }
  uws_server_->Send(msg_data);
  if (msg->web_display_profile_) {
    msg->web_display_profile_->EndRecord();
    LOGD << *msg->web_display_profile_;
  }
  return 0;
}

void WebDisplayModule::map_smart_proc() {
  static uint64_t pre_frame_id = 0;
  while (!map_stop_) {
    std::unique_lock<std::mutex> lock(map_smart_mutex_);
    map_smart_condition_.wait(lock);
    if (j5_frames_.size() > 3 || j5_smart_msg_.size() > 3) {
      LOGD << "map_proc, frames_image size = " << j5_frames_.size()
           << ", smart_msg size = " << j5_smart_msg_.size();
    }
    if (map_stop_) {
      break;
    }
    while (!j5_smart_msg_.empty() && !j5_frames_.empty()) {
      auto msg = j5_smart_msg_.top();
      auto frame = j5_frames_.top();
      if (msg->time_stamp_ == frame->time_stamp_) {
        if (msg->frame_id_ > pre_frame_id ||
            (pre_frame_id - msg->frame_id_ > 300) ||
            pre_frame_id == 0) {  // frame_id maybe overflow reset to 0
          int task_num = data_send_thread_.GetTaskNum();
          if (task_num < 3) {
            data_send_thread_.PostTask(std::bind(
                &WebDisplayModule::SendSmartMessage, this, msg, frame));
          }
          pre_frame_id = msg->frame_id_;
        }
        j5_smart_msg_.pop();
        j5_frames_.pop();
      } else {
        // avoid smart or image result lost
        while (j5_smart_msg_.size() > cache_size_) {
          auto msg_inner = j5_smart_msg_.top();
          auto frame_inner = j5_frames_.top();
          if (msg_inner->time_stamp_ < frame_inner->time_stamp_) {
            // 消息对应的图片一直没有过来，删除消息
            j5_smart_msg_.pop();
          } else {
            break;
          }
        }
        while (j5_frames_.size() > cache_size_) {
          auto msg_inner = j5_smart_msg_.top();
          auto frame_inner = j5_frames_.top();
          if (frame_inner->time_stamp_ < msg_inner->time_stamp_) {
            // 图像对应的消息一直没有过来，删除图像
            j5_frames_.pop();
          } else {
            break;
          }
        }

        break;
      }
    }
    if (j5_smart_msg_.size() > cache_size_) {
      LOGF << "web socket has cache smart message nun > " << cache_size_;
    }
    if (j5_frames_.size() > cache_size_) {
      LOGF << "web socket has cache image nun > " << cache_size_;
    }
  }
}
}  // namespace J5Sample
