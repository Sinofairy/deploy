//
// Created by chuanyi.yang@hobot.cc on 06/15/2018.
// Copyright (c) 2018 horizon robotics. All rights reserved.
//
#ifndef J5_SAMPLE_UTILS_PROFILE_H_
#define J5_SAMPLE_UTILS_PROFILE_H_

#include <chrono>
#include <memory>
#include <mutex>
#include <string>

namespace J5Sample {
struct ModuleProfile {
  explicit ModuleProfile(std::string module_name) {
    module_name_ = module_name;
    module_start_time_ = 0;
    module_end_time_ = 0;
    module_cost_time_ = 0;
    fps_ = 0;
    total_time_ = 0;
    StartRecord();
  }
  std::string module_name_;
  uint64_t module_start_time_;
  uint64_t module_end_time_;
  uint64_t module_cost_time_;
  uint64_t total_time_;
  int32_t fps_;

  inline void StartRecord() {
    auto d = std::chrono::system_clock::now().time_since_epoch();
    std::chrono::milliseconds m =
        std::chrono::duration_cast<std::chrono::milliseconds>(d);
    module_start_time_ = m.count();
    module_end_time_ = module_start_time_;
  }
  inline void EndRecord() {
    auto d = std::chrono::system_clock::now().time_since_epoch();
    std::chrono::milliseconds m =
        std::chrono::duration_cast<std::chrono::milliseconds>(d);
    module_end_time_ = m.count();
    module_cost_time_ = module_end_time_ - module_start_time_;
  }
  inline friend std::ostream &operator<<(std::ostream &out,
                                         ModuleProfile &data) {
    out << "ModuleName: " << data.module_name_
        << ", StartTime: " << data.module_start_time_
        << ", EndTime: " << data.module_end_time_
        << ", Delay(ms): " << data.module_cost_time_;
    return out;
  }
  inline void FrameStatistic() {
    // 实际智能帧率计算
    static int fps = 0;
    // 耗时统计，ms
    static auto last_time = std::chrono::system_clock::now().time_since_epoch();
    static int frame_Count = 0;

    ++frame_Count;

    auto curTime = std::chrono::system_clock::now().time_since_epoch();
    auto val = std::chrono::duration_cast<std::chrono::milliseconds>(curTime -
                                                                     last_time);
    // 统计数据发送帧率
    if (val.count() > 1000) {
      fps = frame_Count;
      frame_Count = 0;
      last_time = std::chrono::system_clock::now().time_since_epoch();
      LOGW << "Sample fps = " << fps;
      fps_ = fps;
    }
  }
};
typedef std::shared_ptr<ModuleProfile> spModuleProfile;
}  // namespace J5Sample
#endif  // J5_SAMPLE_UTILS_PROFILE_H_
