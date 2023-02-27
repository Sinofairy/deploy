/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2021 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/
#ifndef _J5_SAMPLE_INCLUDE_MODULES_CAMERA_INPUT_MODULE_
#define _J5_SAMPLE_INCLUDE_MODULES_CAMERA_INPUT_MODULE_
#include <string>
#include <memory>
#include <mutex>
#include "modules/input/input_module.h"

namespace J5Sample {

struct CameraInputConfig {
  CameraInputConfig() {}
  ~CameraInputConfig() {}
  int cam_index;
  std::string cam_cfg_file;
  int frame_count = -1;
};

class CameraInputModule : public InputModule {
 public:
  CameraInputModule() = delete;
  explicit CameraInputModule(const int &channel_id,
      const std::shared_ptr<InputConfig> input_cfg)
    : InputModule("CameraInputModule", channel_id, input_cfg) {}
  ~CameraInputModule() {}

  virtual int ModuleInit(hobot::RunContext *context);

  virtual int ModuleDeInit();

  virtual int Next(std::shared_ptr<PyramidFrame> &output_image);

  virtual int Free(std::shared_ptr<PyramidFrame> &input_image);

  virtual bool HasNext();

 private:
  int LoadConfig(const std::string &config_file);

 private:
  std::shared_ptr<CameraInputConfig> cam_input_cfg_;
  static std::once_flag cam_init_flag_;
  int count_ = 0;
};

}  // namespace J5Sample
#endif  // _J5_SAMPLE_INCLUDE_MODULES_CAMERA_INPUT_MODULE_
