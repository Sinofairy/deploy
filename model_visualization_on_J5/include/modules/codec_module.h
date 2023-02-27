/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2021 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/
#ifndef _J5_SAMPLE_INCLUDE_MODULES_CODEC_MODULE_
#define _J5_SAMPLE_INCLUDE_MODULES_CODEC_MODULE_

#include <string.h>
#include <vector>
#include <memory>
#include <string>

#include "base_message.h"
#include "hobot/hobot.h"
#include "hobotlog/hobotlog.hpp"
#include "utils/json_cfg_wrapper.h"
#include "libmm/hb_media_codec.h"
#include "libmm/hb_media_error.h"

#define MAX_CODEC_INSTANCE (32)

namespace J5Sample {

struct MediaCodecConfig {
  MediaCodecConfig() {}
  ~MediaCodecConfig() {}
  int chn_id;
  media_codec_id_t codec_id;
  int image_width;
  int image_height;
  int frame_buf_count;
  int jpeg_quality;
};

struct MediaCodecContext {
  MediaCodecContext() {
    context_ = new media_codec_context_t;
    memset(context_, 0, sizeof(media_codec_context_t));
  }
  ~MediaCodecContext() {
    if (context_) delete context_;
  }
  int chn_id_;
  media_codec_context_t *context_;
};

class CodecModule : public hobot::Module {
 public:
  CodecModule() : hobot::Module("J5Sample", "CodecModule") {}
  explicit CodecModule(std::string config_file)
      : hobot::Module("J5Sample", "CodecModule"), config_file_(config_file) {}

  int Init(hobot::RunContext *context) override;

  void Reset() override;

  FORWARD_DECLARE(CodecModule, 0);

 private:
  int LoadConfigFile(const std::string &input_config_file,
      std::shared_ptr<JsonConfigWrapper> &output_json_cfg);
  int LoadConfig(const std::string &config_file);
  int ModuleInit(const std::shared_ptr<MediaCodecConfig> &chn_cfg);
  int ModuleDeInit();
  bool FindAvailableImage(const int &image_width, const int &image_height,
      std::vector<ImageLevelInfo> &image_level, int &output_index);
  bool FindPymLayer(const int &image_width, const int &image_height,
      const std::shared_ptr<PyramidFrame> &input, ImageLevelInfo &output);
  int GetStream(const std::shared_ptr<PyramidFrame> &input,
      std::shared_ptr<StreamFrame> &output);
  int FreeStream(const std::shared_ptr<StreamFrame> &input);
  void GetDumpNum(const std::string &input, bool &dump_en, int &dump_num);
  void DumpInputImage(const std::shared_ptr<PyramidFrame> &frame);
  void DumpOutputImage(const std::shared_ptr<StreamFrame> &stream);

 private:
  std::string config_file_;
  std::vector<std::shared_ptr<MediaCodecConfig>> chn_cfg_list_;
  std::vector<std::shared_ptr<MediaCodecContext>> chn_ctx_list_;
  bool init_flag_ = false;
  int input_dump_count_ = 0;
  int output_dump_count_ = 0;
  std::mutex context_mutex_;
};
}  // namespace J5Sample
#endif  // _J5_SAMPLE_INCLUDE_MODULES_CODEC_MODULE_
