/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2021 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/
#ifndef _J5_SAMPLE_INCLUDE_MODULES_INPUT_MODULE_
#define _J5_SAMPLE_INCLUDE_MODULES_INPUT_MODULE_
#include <string>
#include <memory>
#include <vector>
#include "hobot/hobot.h"
#include "message/base_message.h"
#include "message/vio_message.h"
#include "common/vision_type.h"
#include "utils/json_cfg_wrapper.h"
#include "vio/hb_vpm_data_info.h"
#include "vio/hb_vin_data_info.h"
#include "vio/hb_vio_interface.h"

namespace J5Sample {

struct InputConfig {
  InputConfig() {}
  ~InputConfig() {}
  std::string board_name;
  bool cam_en;
  std::string data_source;
  int data_source_num;
  std::vector<int> channel_id;
  int max_vio_buffer;
  bool is_msg_package;
  std::string source_cfg_file;
  std::string vpm_cfg_file;
};

class InputModule : public hobot::Module {
 public:
  InputModule() = delete;
  explicit InputModule(const std::string &module_name, const int &channel_id,
      const std::shared_ptr<InputConfig> input_cfg)
    : hobot::Module("J5Sample", module_name), module_name_(module_name),
    channel_id_(channel_id), input_cfg_(input_cfg) {}
  ~InputModule() {}

  int Init(hobot::RunContext *context) override;

  void Reset() override;

  FORWARD_DECLARE(InputModule, 0);

  virtual int ModuleInit(hobot::RunContext *context) = 0;

  virtual int ModuleDeInit() = 0;

  virtual int Next(std::shared_ptr<PyramidFrame> &output_image) = 0;

  virtual int Free(std::shared_ptr<PyramidFrame> &input_image) = 0;

  virtual bool HasNext() = 0;

  virtual int NextFrameId() { return ++last_frame_id_; }


 protected:
  int LoadConfigFile(const std::string &input_config_file,
      std::shared_ptr<JsonConfigWrapper> &output_json_cfg);
  int ConvertPymInfo(void* src_buf, void* pym_buf,
      std::shared_ptr<PyramidFrame> pym_img);
  int GetPyramidFrame(void* src_buf,
      std::shared_ptr<PyramidFrame> &output_image);
  int FreePyramidFrame(std::shared_ptr<PyramidFrame> &input_image);
  int DumpToFile2Plane(const char *filename, char *srcBuf, char *srcBuf1,
      unsigned int size, unsigned int size1);
  int DumpToFile(const char *filename, char *srcBuf, unsigned int size);
  void GetDumpNum(const std::string &input, bool &dump_en, int &dump_num);
  void DumpPymBuf(uint32_t pipe_id, uint32_t pym_id,
      void *pym_buffer, int count);

 private:
  bool AllocBuffer();
  void FreeBuffer();
  int GetConsumedBufferNum();

 protected:
  std::string module_name_;
  int channel_id_;
  int pym_id_ = 0;
  std::shared_ptr<InputConfig> input_cfg_;
  bool init_flag_ = false;

 private:
  int last_frame_id_ = 0;
  uint32_t sample_freq_ = 1;
  int consumed_vio_buffers_ = 0;
  std::mutex vio_buffer_mutex_;
  int dump_count_ = 0;
};

}  // namespace J5Sample
#endif  // _J5_SAMPLE_INCLUDE_MODULES_INPUT_MODULE_
