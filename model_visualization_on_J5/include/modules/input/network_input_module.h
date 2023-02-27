/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2021 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/
#ifndef _J5_SAMPLE_INCLUDE_MODULES_NETWORK_INPUT_MODULE_
#define _J5_SAMPLE_INCLUDE_MODULES_NETWORK_INPUT_MODULE_
#include <string>
#include <memory>
#include "modules/input/input_module.h"
#include "modules/input/network_receiver.h"

namespace J5Sample {

struct NetworkInputConfig {
  NetworkInputConfig() {}
  ~NetworkInputConfig() {}
  std::string ip_port;
  HorizonVisionPixelFormat data_type;
  int height;
  int width;
  bool with_image_name;
};

class NetworkInputModule : public InputModule {
 public:
  NetworkInputModule() = delete;
  explicit NetworkInputModule(const int &channel_id,
      const std::shared_ptr<InputConfig> input_cfg)
    : InputModule("NetworkInputModule", channel_id, input_cfg) {}
  ~NetworkInputModule() {}

  virtual int ModuleInit(hobot::RunContext *context);

  virtual int ModuleDeInit();

  virtual int Next(std::shared_ptr<PyramidFrame> &output_image);

  virtual int Free(std::shared_ptr<PyramidFrame> &input_image);

  virtual bool HasNext();

 private:
  int LoadConfig(const std::string &config_file);

 private:
  std::shared_ptr<NetworkInputConfig> network_input_cfg_;
  NetworkReceiver receiver_;
  static std::once_flag fb_init_flag_;
  bool is_finish_ = false;
  int dump_count_ = 0;
};




}  // namespace J5Sample
#endif  // _J5_SAMPLE_INCLUDE_MODULES_NETWORK_INPUT_MODULE_
