/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2021 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/
#include <memory>
#include <string>
#include <fstream>
#include <vector>
#include "modules/vio_module.h"
#include "modules/input/camera_input_module.h"
#include "modules/input/network_input_module.h"
#include "hobotlog/hobotlog.hpp"

namespace J5Sample {

std::once_flag VioModule::vio_flag_;
std::shared_ptr<VioModule> VioModule::vio_worker_;

int VioModule::LoadConfig(const std::string config_file) {
  int config_index;
  LOGI << "Load vio config file:" << config_file;
  std::ifstream ifs(config_file);
  if (!ifs) {
    LOGE << "error!!! vio config file is not exist";
    return -1;
  }
  Json::Value cfg_jv;
  std::shared_ptr<JsonConfigWrapper> vio_cfg_json = nullptr;
  ifs >> cfg_jv;
  vio_cfg_json.reset(new JsonConfigWrapper(cfg_jv));

  input_cfg_ = std::make_shared<InputConfig>();
  if (vio_cfg_json->HasMember("config_index")) {
    config_index = vio_cfg_json->GetIntValue("config_index");
  } else {
    LOGE << "config index parameter is not exist";
    return -1;
  }
  if (vio_cfg_json->HasMember("board_name")) {
    input_cfg_->board_name = vio_cfg_json->GetSTDStringValue("board_name");
  }

  std::string config_index_name = "config_" + std::to_string(config_index);
  auto config_json = vio_cfg_json->GetSubConfig(config_index_name);
  HOBOT_CHECK(config_json);

  // parse input config
  if (config_json->HasMember("cam_en")) {
    input_cfg_->cam_en = config_json->GetBoolValue("cam_en");
  }
  if (config_json->HasMember("data_source")) {
    input_cfg_->data_source = config_json->GetSTDStringValue("data_source");
  }
  if (config_json->HasMember("data_source_num")) {
    input_cfg_->data_source_num = config_json->GetIntValue("data_source_num");
  }
  if (config_json->HasMember("channel_id")) {
    input_cfg_->channel_id = config_json->GetIntArray("channel_id");
  }
  if (config_json->HasMember("max_vio_buffer")) {
    input_cfg_->max_vio_buffer = config_json->GetIntValue("max_vio_buffer");
  }
  if (config_json->HasMember("is_msg_package")) {
    input_cfg_->is_msg_package = config_json->GetIntValue("is_msg_package");
  }
  if (config_json->HasMember("source_cfg_file")) {
    input_cfg_->source_cfg_file =
      config_json->GetSTDStringValue("source_cfg_file");
  }
  if (config_json->HasMember("vpm_cfg_file")) {
    input_cfg_->vpm_cfg_file =
      config_json->GetSTDStringValue("vpm_cfg_file");
  }

  int data_source_num = input_cfg_->data_source_num;
  int channel_id_num = input_cfg_->channel_id.size();
  HOBOT_CHECK(data_source_num == channel_id_num)
    << "data_source_num: " << data_source_num
    << " channel_id_num: " << channel_id_num;

  LOGI  << "board_name: " << input_cfg_->board_name;
  LOGI  << "cam_en: " << input_cfg_->cam_en;
  LOGI  << "data_source: " << input_cfg_->data_source;
  LOGI  << "data_source_num: " << input_cfg_->data_source_num;
  LOGI  << "max_vio_buffer: " << input_cfg_->max_vio_buffer;
  LOGI  << "source_cfg_file:" << input_cfg_->source_cfg_file;
  LOGI  << "vpm_cfg_file:" << input_cfg_->vpm_cfg_file;
  return 0;
}

std::shared_ptr<VioModule> VioModule::GetInstance() {
  if (!vio_worker_) {
    std::call_once(vio_flag_,
        [&]() {
        vio_worker_ = std::make_shared<VioModule>();
        });
  }
  return vio_worker_;
}

std::vector<std::shared_ptr<InputModule>> VioModule::\
    GetInputModuleHandle(const std::string &config_file) {
  int ret = -1;
  std::shared_ptr<InputModule> input_module;
  std::vector<std::shared_ptr<InputModule>> tmp;

  ret = LoadConfig(config_file);
  HOBOT_CHECK(ret == 0);
  int chn_num = input_cfg_->data_source_num;
  for (int chn_index = 0; chn_index < chn_num; chn_index++) {
    if (input_cfg_->data_source == "mipi_camera") {
      input_module = std::make_shared<CameraInputModule>(
          chn_index, input_cfg_);
    } else if (input_cfg_->data_source == "network_feedback") {
      input_module = std::make_shared<NetworkInputModule>(
          chn_index, input_cfg_);
    } else {
      LOGE << "Unsupport data source: " << input_cfg_->data_source;
      return tmp;
    }
    AddTask();
    input_module_list_.push_back(input_module);
  }
  return input_module_list_;
}

}  // namespace J5Sample

