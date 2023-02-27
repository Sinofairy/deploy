/**
 * Copyright Horizon Robotics
 */
#ifndef _J5_SAMPLE_INCLUDE_MODULES_SAMPLE_OBSERVER_MODULE_
#define _J5_SAMPLE_INCLUDE_MODULES_SAMPLE_OBSERVER_MODULE_

#include <iostream>

#include "hobot/hobot.h"
#include "hobotlog/hobotlog.hpp"

namespace J5Sample {
class SampleObserver : public hobot::RunObserver {
 public:
  void OnResult(hobot::Module *from, int forward_index,
                hobot::spMessage output) override {
    LOGD << "SampleObserver::result:" << std::endl;
  }
};
}  // namespace J5Sample
#endif  // _J5_SAMPLE_INCLUDE_MODULES_SAMPLE_OBSERVER_MODULE_

