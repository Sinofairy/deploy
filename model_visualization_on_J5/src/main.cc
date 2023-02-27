/**
 * Copyright Horizon Robotics
 */
#include <thread>

#include "hobot/hobot.h"
#include "hobotlog/hobotlog.hpp"
#include "modules/codec_module.h"
#include "modules/inference_module.h"
#include "modules/sample_observer.h"
#include "modules/vio_module.h"
#include "modules/web_display_module.h"

using hobot::spMessage;
using J5Sample::CodecModule;
using J5Sample::InferenceModule;
using J5Sample::J5FrameMessage;
using J5Sample::SampleObserver;
using J5Sample::VioModule;
using J5Sample::WebDisplayModule;

typedef std::shared_ptr<hobot::Module> spModule;
static bool exit_ = false;

static void signal_handle(int param) {
  std::cout << "recv signal " << param << ", stop" << std::endl;
  if (param == SIGINT) {
    exit_ = true;
  }
}

int main(int argc, char **argv) {
  std::string log_level;
  if (argv[1] == nullptr) {
    std::cout << "set default log level: [-i] ";
    log_level = "-i";
  } else {
    log_level = argv[1];
  }
  if (log_level == "-i") {
    SetLogLevel(HOBOT_LOG_INFO);
  } else if (log_level == "-d") {
    SetLogLevel(HOBOT_LOG_DEBUG);
  } else if (log_level == "-w") {
    SetLogLevel(HOBOT_LOG_WARN);
  } else if (log_level == "-e") {
    SetLogLevel(HOBOT_LOG_ERROR);
  } else if (log_level == "-f") {
    SetLogLevel(HOBOT_LOG_FATAL);
  } else {
    std::cout << "set default log level: [-i] " << std::endl;
    SetLogLevel(HOBOT_LOG_INFO);
  }
  LOGD << "Start Sample.";
  hobot::Engine *engine = hobot::Engine::NewInstance();
  hobot::Workflow *workflow = engine->NewWorkflow();

  signal(SIGINT, signal_handle);
  signal(SIGPIPE, signal_handle);

  auto vio_module_inst = VioModule::GetInstance();
  std::string vio_config_file = "./configs/vio/j5_vio_config.json";
  auto input_module_handle_list =
      vio_module_inst->GetInputModuleHandle(vio_config_file);
  HOBOT_CHECK(input_module_handle_list.size() > 0);
  // only test single vio module
  spModule vio_module =
      std::static_pointer_cast<hobot::Module>(input_module_handle_list[0]);

  std::string codec_config_file = "./configs/codec/j5_codec_config.json";
  spModule codec_module = std::make_shared<CodecModule>(codec_config_file);

  std::string inference_config_file = "./configs/model/inference_config.json";
  spModule inference_module =
      std::make_shared<InferenceModule>(inference_config_file);
  spModule web_display_module = std::make_shared<WebDisplayModule>("");

  workflow->From(vio_module.get())->To(inference_module.get(), 0);
  workflow->From(vio_module.get())->To(codec_module.get(), 0);
  workflow->From(inference_module.get())->To(web_display_module.get(), 0);
  workflow->From(codec_module.get())->To(web_display_module.get(), 1);

  engine->ExecuteOnThread(vio_module.get(), 0, 0);
  engine->ExecuteOnThread(inference_module.get(), 0, 1);
  engine->ExecuteOnThread(web_display_module.get(), 0, 2);
  engine->ExecuteOnThread(codec_module.get(), 0, 3);

  std::shared_ptr<SampleObserver> out = std::make_shared<SampleObserver>();
  hobot::spRunContext run_task =
      workflow->Run({std::make_pair(web_display_module.get(), 0)}, out.get());
  run_task->Init();
  spMessage input_frame = std::make_shared<J5FrameMessage>();
  workflow->Feed(run_task, vio_module.get(), 0, input_frame);
  LOGD << "Feed input trigger frame.";
  // sleep
  while (!exit_) {
    std::this_thread::sleep_for(std::chrono::microseconds(60));
  }

  workflow->Reset();
  LOGD << "Sample End";
  delete workflow;
  delete engine;

  return 0;
}

