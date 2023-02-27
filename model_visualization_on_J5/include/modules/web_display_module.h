/**
 * Copyright Horizon Robotics
 */

#ifndef _J5_SAMPLE_INCLUDE_MODULES_WEB_DISPLAY_MODULE_
#define _J5_SAMPLE_INCLUDE_MODULES_WEB_DISPLAY_MODULE_

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "base_message.h"
#include "hobot/hobot.h"
#include "hobotlog/hobotlog.hpp"
#include "thread_pool.h"
#include "uws_server/uws_server.h"

namespace J5Sample {
using J5Sample::CThreadPool;
using J5Sample::J5FrameMessage;

struct CodecData {
 public:
  CodecData() {}
  ~CodecData() {}

  int channel_ = -1;
  // frame id
  uint64_t frame_id_ = 0;
  // time stamp
  uint64_t time_stamp_ = 0;
  // codec image data
  uint16_t src_width_;
  uint16_t src_height_;
  uint32_t stream_size_;
  uint64_t stream_pts_;
  uint64_t stream_paddr_;
  uint64_t stream_vaddr_;
};
typedef std::shared_ptr<CodecData> spCodecData;

struct CompareFrame {
  bool operator()(const spCodecData f1, const spCodecData f2) {
    return (f1->time_stamp_ > f2->time_stamp_);
  }
};

struct CompareMsg {
  bool operator()(const spJ5FrameMessage m1, const spJ5FrameMessage m2) {
    return (m1->time_stamp_ > m2->time_stamp_);
  }
};

class WebDisplayModule : public hobot::Module {
 public:
  WebDisplayModule() : hobot::Module("J5Sample", "WebDisplayModule") {}
  explicit WebDisplayModule(std::string config_file)
      : hobot::Module("J5Sample", "WebDisplayModule"),
        config_file_(config_file) {}
  int Init(hobot::RunContext *context) override;

  void Reset() override;

  FORWARD_DECLARE(WebDisplayModule, 0);

 private:
  int SendSmartMessage(spJ5FrameMessage msg, spCodecData codec_frame);
  std::string Serilize(spJ5FrameMessage msg, spCodecData codec_frame);
  void map_smart_proc();

 private:
  std::string config_file_;
  int dst_image_width_ = 1920;
  int dst_image_height_ = 1080;
  CThreadPool data_send_thread_;
  std::shared_ptr<std::thread> worker_;
  std::shared_ptr<UwsServer> uws_server_;
  std::mutex map_smart_mutex_;
  bool map_stop_ = false;
  std::condition_variable map_smart_condition_;
  const uint8_t cache_size_ = 20;  // max input cache size
  // channel_id, frame;
  std::priority_queue<spCodecData, std::vector<spCodecData>, CompareFrame>
      j5_frames_;
  // channel_id, smart_result
  std::priority_queue<spJ5FrameMessage, std::vector<spJ5FrameMessage>,
                      CompareMsg>
      j5_smart_msg_;
  std::shared_ptr<ModuleProfile> profile_ = nullptr;
};
}  // namespace J5Sample
#endif  // _J5_SAMPLE_INCLUDE_MODULES_WEB_DISPLAY_MODULE_
