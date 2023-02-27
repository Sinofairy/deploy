/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2021 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/
#ifndef _J5_SAMPLE_INCLUDE_MODULES_VIO_MODULE_
#define _J5_SAMPLE_INCLUDE_MODULES_VIO_MODULE_
#include <string>
#include <memory>
#include <vector>
#include <mutex>
#include <atomic>
#include "modules/input/input_module.h"

namespace J5Sample {

class VioModule {
 public:
  static std::shared_ptr<VioModule> GetInstance();
  VioModule() = default;
  ~VioModule() {
    input_module_list_.clear();
  }

  std::vector<std::shared_ptr<InputModule>> GetInputModuleHandle(
      const std::string &config_file);
  int GetTaskNum() { return task_num_; }
  void AddTask() { task_num_++; }
  void FreeTask() { task_num_--; }

 private:
  int LoadConfig(const std::string config_file);

 private:
  static std::once_flag vio_flag_;
  static std::shared_ptr<VioModule> vio_worker_;
  std::vector<std::shared_ptr<InputModule>> input_module_list_;
  std::shared_ptr<InputConfig> input_cfg_;
  std::atomic_int task_num_{0};
};

}  // namespace J5Sample
#endif  // _J5_SAMPLE_INCLUDE_MODULES_VIO_MODULE_
