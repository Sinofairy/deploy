/**
 * Copyright Horizon Robotics
 */
#include "inference_module.h"

#include <fstream>

#include "hobotlog/hobotlog.hpp"
#include "image_utils.h"

namespace J5Sample {
#define HB_CHECK_SUCCESS(value, errmsg)             \
  do {                                               \
    /*value can be call of function*/                \
    auto ret_code = value;                           \
    if (ret_code != 0) {                             \
      LOGE << errmsg << ", error code:" << ret_code; \
      return ret_code;                               \
    }                                                \
  } while (0);

int InferenceModule::LoadConfig() {
  std::string default_value = "";
  LOGI << "Load Inference config file:" << config_file_;
  Json::Value cfg_jv;
  std::ifstream infile(config_file_);
  if (!infile) {
    LOGE << "error!!! Inference config file is not exist";
    return -1;
  }
  infile >> cfg_jv;
  config_.reset(new JsonConfigWrapper(cfg_jv));

  if (config_->HasMember("model_file_path")) {
    model_file_path_ = config_->GetSTDStringValue("model_file_path");
  } else {
    LOGE << "config has not model_file_path";
    return -1;
  }
  if (config_->HasMember("pyramid_level")) {
    pym_level_ = config_->GetIntValue("pyramid_level");
  } else {
    LOGE << "config has not pyramid_level";
    return -1;
  }
  if (config_->HasMember("model_detect_class")) {
    auto model_class_config = config_->GetSubConfig("model_detect_class");
    auto classify_names = model_class_config->GetJsonKeys();
    for (auto name : classify_names) {
      auto id = model_class_config->GetIntValue(name);
      detect_classify_[id] = name;
    }
  } else {
    LOGE << "config has not model_detect_class";
    return -1;
  }
  LOGD << "Load config suc";
  return 0;
}

int InferenceModule::Init(hobot::RunContext *context) {
  // load model
  LOGD << "Enter InferenceModule::Init";
  HB_CHECK_SUCCESS(LoadConfig(), "LoadConfig failed.");
  auto modelFileName = model_file_path_.c_str();
  HB_CHECK_SUCCESS(
      hbDNNInitializeFromFiles(&packed_dnn_handle_, &modelFileName, 1),
      "hbDNNInitializeFromFiles failed");
  LOGD << "hbDNNInitializeFromFiles success";

  // get dnn handle
  const char **model_name_list;
  int model_count = 0;
  HB_CHECK_SUCCESS(
      hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle_),
      "hbDNNGetModelNameList failed");
  LOGD << "hbDNNGetModelNameList success";
  HOBOT_CHECK(model_name_list[0]) << "model name is invalid.";

  // get model handle
  HB_CHECK_SUCCESS(
      hbDNNGetModelHandle(&dnn_handle_, packed_dnn_handle_, model_name_list[0]),
      "hbDNNGetModelHandle failed");
  LOGD << "hbDNNGetModelHandle success";
  HB_CHECK_SUCCESS(
      hbDNNGetInputTensorProperties(&input_properties_, dnn_handle_, 0),
      "hbDNNGetInputTensorProperties failed");

  int h_idx, w_idx, c_idx;
  get_hwc_index(input_properties_.tensorLayout, &h_idx, &w_idx, &c_idx);
  model_intput_width_ = input_properties_.validShape.dimensionSize[w_idx];
  model_intput_height_ = input_properties_.validShape.dimensionSize[h_idx];
  worker_ = std::make_shared<CThreadPool>();
  worker_->CreatThread(3);
  LOGD << "Leave InferenceModule::Init";
  is_inited_ = true;
  return 0;
}

spVioMessage InferenceModule::TestCreateVioMessage() {
  static int time_id = 0;
  cv::Mat yuv_mat;
  std::string test_pic = "./configs/data/000000.png";
  if (0 != read_image_2_nv12(test_pic, 540, 960, yuv_mat)) {
    LOGE << "read_image_2_nv12 failed";
  }
  int data_len = 960 * 540 * 3 / 2;
  uint8_t *pbase = reinterpret_cast<uint8_t *>(std::calloc(1, data_len));
  memcpy(pbase, yuv_mat.ptr<uint8_t>(), data_len);
  auto vio_message = std::make_shared<VioMessage>();
  spPyramidFrame pym_frame(new PyramidFrame(), [&](PyramidFrame *p) {
    if (p) {
      if (p->bl_ds_.size() > 0) {
        LOGD << "delete rawdata image frame";
        std::free(reinterpret_cast<uint8_t *>(p->bl_ds_[0].y_vaddr));
        p->bl_ds_.clear();
      }
      delete (p);
      p = nullptr;
    }
  });
  vio_message->pym_image_ = pym_frame;
  vio_message->channel_ = 1;
  vio_message->frame_id_ = 1;
  vio_message->time_stamp_ = time_id++;
  ImageLevelInfo image_info;
  image_info.width = 960;
  image_info.height = 540;
  image_info.stride = 960;
  image_info.y_vaddr = reinterpret_cast<uint64_t>(pbase);
  image_info.c_vaddr = reinterpret_cast<uint64_t>(pbase + 960 * 540);
  pym_frame->bl_ds_.push_back(image_info);
  pym_frame->src_info_.width = 1920;  //  test
  pym_frame->src_info_.height = 1080;
  return vio_message;
}

FORWARD_DEFINE(InferenceModule, 0) {
  auto vio_message = std::dynamic_pointer_cast<VioMessage>((*input[0])[0]);
  // do something
  if (nullptr != vio_message) {
    worker_->PostTask(std::bind(&InferenceModule::DoProcess, this, vio_message,
                              workflow, this, 0, context));
  } else {
    LOGD << "VioMessage is nullptr.";
    workflow->Return(this, 0, nullptr, context);
  }
}

void InferenceModule::Reset() {
  LOGD << "Enter InferenceModule::Reset";
  if (false == is_inited_) {
    return;
  }
  is_inited_ = false;
  worker_->ClearTask();
  worker_ = nullptr;
  // release model
  if (packed_dnn_handle_) {
    if (0 != hbDNNRelease(packed_dnn_handle_)) {
      LOGE << "hbDNNRelease failed";
    }
  }
  LOGD << "leave InferenceModule::Reset";
}

int InferenceModule::DoProcess(spVioMessage pyramid_message,
                               hobot::Workflow *workflow, Module *from,
                               int output_slot, hobot::spRunContext context) {
  ModuleProfile profile("InferenceModule");
  // prepare out message
  auto out_message = std::make_shared<J5FrameMessage>();
  out_message->channel_id_ = pyramid_message->channel_;
  out_message->time_stamp_ = pyramid_message->time_stamp_;
  out_message->frame_id_ = pyramid_message->frame_id_;
  // prepare tensor
  int input_count = 0;
  int output_count = 0;
  hbDNNTensor *input_tensor = nullptr;
  hbDNNTensor *output_tensor = nullptr;

  hbDNNGetInputCount(&input_count, dnn_handle_);
  hbDNNGetOutputCount(&output_count, dnn_handle_);
  input_tensor = new hbDNNTensor[input_count];
  output_tensor = new hbDNNTensor[output_count];
  HB_CHECK_SUCCESS(prepare_input_tensor(input_tensor, input_count),
                    "prepare input tensor failed");
  HB_CHECK_SUCCESS(prepare_output_tensor(&output_tensor, output_count),
                    "prepare output tensor failed");
  // prepare for input
  int src_image_width = pyramid_message->pym_image_->src_info_.width;
  int src_image_height = pyramid_message->pym_image_->src_info_.height;
  int width = pyramid_message->pym_image_->bl_ds_[0].width;
  int height = pyramid_message->pym_image_->bl_ds_[0].height;
  uint8_t *y_data = reinterpret_cast<uint8_t *>(
      pyramid_message->pym_image_->bl_ds_[0].y_vaddr);
  uint8_t *c_data = reinterpret_cast<uint8_t *>(
      pyramid_message->pym_image_->bl_ds_[0].c_vaddr);
  HB_CHECK_SUCCESS(
      prepare_nv12_tensor(input_tensor, y_data, c_data, height, width),
      "prepare nv12 tensor failed");
  pyramid_message = nullptr;
  LOGD << "prepare nv12 tensor success";

  // Run inference
  hbDNNTaskHandle_t task_handle = nullptr;
  hbDNNInferCtrlParam infer_ctrl_param;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
  HB_CHECK_SUCCESS(hbDNNInfer(&task_handle, &output_tensor, input_tensor,
                              dnn_handle_, &infer_ctrl_param),
                   "hbDNNInfer failed");
  LOGD << "infer success";

  // wait task done
  HB_CHECK_SUCCESS(hbDNNWaitTaskDone(task_handle, 0),
                   "hbDNNWaitTaskDone failed");
  LOGD << "task done";
  // get result
  get_bbox_result(&output_tensor[1], out_message);
  get_seg_result(&output_tensor[2], out_message);
  CoordinateTransform(out_message, src_image_width, src_image_height,
                      model_intput_width_, model_intput_height_);
  LOGD << "task post process finished";

  // release task handle
  HB_CHECK_SUCCESS(hbDNNReleaseTask(task_handle), "hbDNNReleaseTask failed");
  task_handle = nullptr;
  // free input mem
  if (0 != hbSysFreeMem(&(input_tensor->sysMem[0]))) {
    LOGE << "hbSysFreeMem failed";
  }
  if (0 != hbSysFreeMem(&(input_tensor->sysMem[1]))) {
    LOGE << "hbSysFreeMem failed";
  }
  delete[] input_tensor;
  // free output mem
  for (int i = 0; i < output_count; i++) {
    if (0 != hbSysFreeMem(&(output_tensor[i].sysMem[0]))) {
      LOGE << "hbSysFreeMem failed";
    }
  }
  delete[] output_tensor;
  profile.EndRecord();
  LOGD << profile;
  LOGI << "InferenceModule.Forward0 return";
  workflow->Return(from, output_slot, out_message, context);
  return 0;
}

int InferenceModule::get_hwc_index(int32_t layout, int *h_idx,
                                   int *w_idx, int *c_idx) {
  if (layout == HB_DNN_LAYOUT_NHWC) {
    *h_idx = 1;
    *w_idx = 2;
    *c_idx = 3;
  } else if (layout == HB_DNN_LAYOUT_NCHW) {
    *c_idx = 1;
    *h_idx = 2;
    *w_idx = 3;
  }
  return 0;
}

int InferenceModule::padding_nv12_image(uint8_t **pOutputImg, void *yuv_data,
                                        void *uv_data, int image_height,
                                        int image_width) {
  int output_img_size = 0;
  int first_stride = 0;
  int second_stride = 0;
  int output_img_width = 0;
  int output_img_height = 0;

  int y_size = image_height * image_width;
  int uv_size = y_size >> 1;

  const uint8_t *input_nv12_data[3] = {reinterpret_cast<uint8_t *>(yuv_data),
                                       reinterpret_cast<uint8_t *>(uv_data),
                                       nullptr};
  const int input_nv12_size[3] = {y_size, uv_size, 0};
  auto ret = HobotXStreamCropYuvImageWithPaddingBlack(
      input_nv12_data, input_nv12_size, image_width, image_height, image_width,
      image_width, IMAGE_TOOLS_RAW_YUV_NV12, 0, 0, model_intput_width_ - 1,
      model_intput_height_ - 1, pOutputImg, &output_img_size, &output_img_width,
      &output_img_height, &first_stride, &second_stride);

  if (ret < 0) {
    LOGE << "fail to crop image";
    free(*pOutputImg);
    return -1;
  }
  HOBOT_CHECK_EQ(output_img_height, model_intput_height_)
      << "padding error, output_img_height not eq model_input_height";
  HOBOT_CHECK_EQ(output_img_width, model_intput_width_)
      << "padding error, output_img_width not eq model_intput_width";
  return 0;
}

int InferenceModule::prepare_input_tensor(hbDNNTensor *tensor,
                                          int intput_count) {
  tensor->properties = input_properties_;
  auto &tensor_property = tensor->properties;
  int32_t image_data_type = tensor_property.tensorType;
  int32_t layout = tensor_property.tensorLayout;
  int h_idx, w_idx, c_idx;
  get_hwc_index(layout, &h_idx, &w_idx, &c_idx);
  int height = tensor_property.validShape.dimensionSize[h_idx];
  int width = tensor_property.validShape.dimensionSize[w_idx];
  int stride = tensor_property.alignedShape.dimensionSize[w_idx];
  LOGD << "layout = " << width << ", " << height << ", " << stride;
  if (image_data_type == HB_DNN_IMG_TYPE_Y) {
    hbSysAllocCachedMem(&tensor->sysMem[0], height * stride);
  } else if (image_data_type == HB_DNN_IMG_TYPE_YUV444 ||
             image_data_type == HB_DNN_IMG_TYPE_BGR ||
             image_data_type == HB_DNN_IMG_TYPE_RGB) {
    hbSysAllocCachedMem(&tensor->sysMem[0], height * width * 3);
  } else if (image_data_type == HB_DNN_IMG_TYPE_NV12) {
    int y_length = height * stride;
    int uv_length = height / 2 * stride;
    hbSysAllocCachedMem(&tensor->sysMem[0], y_length + uv_length);
  } else if (image_data_type == HB_DNN_IMG_TYPE_NV12_SEPARATE) {
    int y_length = height * stride;
    int uv_length = height / 2 * stride;
    hbSysAllocCachedMem(&tensor->sysMem[0], y_length);
    hbSysAllocCachedMem(&tensor->sysMem[1], uv_length);
  } else if (image_data_type == HB_DNN_TENSOR_TYPE_F32 ||
             image_data_type == HB_DNN_TENSOR_TYPE_S32 ||
             image_data_type == HB_DNN_TENSOR_TYPE_U32) {
    hbSysAllocCachedMem(&tensor->sysMem[0], height * width * 4);
  } else if (image_data_type == HB_DNN_TENSOR_TYPE_U8 ||
             image_data_type == HB_DNN_TENSOR_TYPE_S8) {
    hbSysAllocCachedMem(&tensor->sysMem[0], height * width);
  }
  return 0;
}

void InferenceModule::prepare_image_data(void *image_data,
                                         hbDNNTensor *tensor) {
  auto &tensor_property = tensor->properties;
  int32_t image_data_type = tensor_property.tensorType;
  int32_t layout = tensor_property.tensorLayout;
  int h_idx, w_idx, c_idx;
  get_hwc_index(layout, &h_idx, &w_idx, &c_idx);
  int image_height = tensor_property.validShape.dimensionSize[h_idx];
  int image_width = tensor_property.validShape.dimensionSize[w_idx];
  int stride = tensor_property.alignedShape.dimensionSize[w_idx];
  LOGD << "layout = " << image_width << ", " << image_height << ", " << stride;
  if (image_data_type == HB_DNN_IMG_TYPE_Y) {
    uint8_t *data = reinterpret_cast<uint8_t *>(image_data);
    uint8_t *data0 = reinterpret_cast<uint8_t *>(tensor->sysMem[0].virAddr);
    for (int h = 0; h < image_height; ++h) {
      auto *raw = data0 + h * stride;
      for (int w = 0; w < image_width; ++w) {
        *raw++ = *data++;
      }
    }
    hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_CLEAN);
  } else if (image_data_type == HB_DNN_IMG_TYPE_YUV444 ||
             image_data_type == HB_DNN_IMG_TYPE_BGR ||
             image_data_type == HB_DNN_IMG_TYPE_RGB) {
    if (layout == HB_DNN_LAYOUT_NHWC) {
      uint8_t *data = reinterpret_cast<uint8_t *>(image_data);
      int image_length = image_height * stride * 3;
      void *data0 = tensor->sysMem[0].virAddr;
      memcpy(data0, data, image_length);
    } else {
      int channel_size = image_height * image_width;
      uint8_t *mem = reinterpret_cast<uint8_t *>(tensor->sysMem[0].virAddr);
      nhwc_to_nchw(mem, mem + channel_size, mem + channel_size * 2,
                   reinterpret_cast<uint8_t *>(image_data), image_height,
                   image_width);
    }
    hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_CLEAN);
  } else if (image_data_type == HB_DNN_IMG_TYPE_NV12) {
    uint8_t *data = reinterpret_cast<uint8_t *>(image_data);
    // Align by 16 bytes
    int y_length = image_height * stride;

    // Copy y data to data
    uint8_t *y = reinterpret_cast<uint8_t *>(tensor->sysMem[0].virAddr);
    for (int h = 0; h < image_height; ++h) {
      auto *raw = y + h * stride;
      for (int w = 0; w < image_width; ++w) {
        *raw++ = *data++;
      }
    }
    // Copy uv data
    uint8_t *uv =
        reinterpret_cast<uint8_t *>(tensor->sysMem[0].virAddr) + y_length;
    int uv_height = image_height / 2;
    for (int i = 0; i < uv_height; ++i) {
      auto *raw = uv + i * stride;
      for (int j = 0; j < image_width; ++j) {
        *raw++ = *data++;
      }
    }
    hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_CLEAN);
  } else if (image_data_type == HB_DNN_IMG_TYPE_NV12_SEPARATE) {
    uint8_t *data = reinterpret_cast<uint8_t *>(image_data);

    // Copy y data to data0
    uint8_t *y = reinterpret_cast<uint8_t *>(tensor->sysMem[0].virAddr);
    for (int h = 0; h < image_height; ++h) {
      auto *raw = y + h * stride;
      for (int w = 0; w < image_width; ++w) {
        *raw++ = *data++;
      }
    }
    hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_CLEAN);
    // Copy uv data to data_ext
    uint8_t *uv = reinterpret_cast<uint8_t *>(tensor->sysMem[1].virAddr);
    int uv_height = image_height / 2;
    for (int i = 0; i < uv_height; ++i) {
      auto *raw = uv + i * stride;
      for (int j = 0; j < image_width; ++j) {
        *raw++ = *data++;
      }
    }
    hbSysFlushMem(&(tensor->sysMem[1]), HB_SYS_MEM_CACHE_CLEAN);
  }
  return;
}

void InferenceModule::nhwc_to_nchw(uint8_t *out_data0, uint8_t *out_data1,
                                   uint8_t *out_data2, uint8_t *in_data,
                                   int height, int width) {
  for (int hh = 0; hh < height; ++hh) {
    for (int ww = 0; ww < width; ++ww) {
      *out_data0++ = *(in_data++);
      *out_data1++ = *(in_data++);
      *out_data2++ = *(in_data++);
    }
  }
}

void InferenceModule::release_tensor(hbDNNTensor *tensor) {
  switch (tensor->properties.tensorType) {
    case HB_DNN_IMG_TYPE_Y:
    case HB_DNN_IMG_TYPE_RGB:
    case HB_DNN_IMG_TYPE_BGR:
    case HB_DNN_IMG_TYPE_YUV444:
    case HB_DNN_IMG_TYPE_NV12:
      hbSysFreeMem(&(tensor->sysMem[0]));
      break;
    case HB_DNN_IMG_TYPE_NV12_SEPARATE:
      hbSysFreeMem(&(tensor->sysMem[0]));
      hbSysFreeMem(&(tensor->sysMem[1]));
      break;
    default:
      break;
  }
}

int InferenceModule::prepare_nv12_tensor(hbDNNTensor *input_tensor,
                                         void *yuv_data, void *uv_data,
                                         int image_height, int image_width) {
  uint8_t *pOutputImg = nullptr;

  HB_CHECK_SUCCESS(padding_nv12_image(&pOutputImg, yuv_data, uv_data,
                                       image_height, image_width),
                    "image padding failed.");
  prepare_image_data(pOutputImg, input_tensor);
  free(pOutputImg);
  return 0;
}

int InferenceModule::read_image_2_nv12(std::string &image_file,
                                       int32_t yuv_height, int32_t yuv_width,
                                       cv::Mat &img_nv12) {
  cv::Mat bgr_mat = cv::imread(image_file, cv::IMREAD_COLOR);
  if (bgr_mat.empty()) {
    LOGE << "image file not exist!";
    return -1;
  }
  // resize
  cv::Mat mat;
  mat.create(yuv_height, yuv_width, bgr_mat.type());
  cv::resize(bgr_mat, mat, mat.size(), 0, 0);

  auto ret = bgr_to_nv12(mat, img_nv12);
  return ret;
}

int InferenceModule::prepare_output_tensor(hbDNNTensor **output_tensor,
                                           int output_count) {
  hbDNNTensor *output = *output_tensor;
  for (int i = 0; i < output_count; i++) {
    hbDNNTensorProperties &output_properties = output[i].properties;
    HB_CHECK_SUCCESS(
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle_, i),
        "hbDNN_getOutputTensorProperties failed");

    // get all aligned size
    int out_aligned_size = 1;
    if (output_properties.tensorType >= HB_DNN_TENSOR_TYPE_F16 &&
        output_properties.tensorType <= HB_DNN_TENSOR_TYPE_U16) {
      out_aligned_size = 2;
    } else if (output_properties.tensorType >= HB_DNN_TENSOR_TYPE_F32 &&
               output_properties.tensorType <= HB_DNN_TENSOR_TYPE_U32) {
      out_aligned_size = 4;
    } else if (output_properties.tensorType >= HB_DNN_TENSOR_TYPE_F64 &&
               output_properties.tensorType <= HB_DNN_TENSOR_TYPE_U64) {
      out_aligned_size = 8;
    }
    for (int j = 0; j < output_properties.alignedShape.numDimensions; j++) {
      out_aligned_size =
          out_aligned_size * output_properties.alignedShape.dimensionSize[j];
    }
    out_aligned_size = ((out_aligned_size + (16 - 1)) / 16 * 16);
    hbSysMem &mem = output[i].sysMem[0];
    HB_CHECK_SUCCESS(hbSysAllocCachedMem(&mem, out_aligned_size),
                     "hbSysAllocCachedMem failed");
  }

  return 0;
}

int32_t InferenceModule::bgr_to_nv12(cv::Mat &bgr_mat, cv::Mat &img_nv12) {
  auto height = bgr_mat.rows;
  auto width = bgr_mat.cols;

  if (height % 2 || width % 2) {
    LOGE << "input img height and width must aligned by 2!";
    return -1;
  }
  cv::Mat yuv_mat;
  cv::cvtColor(bgr_mat, yuv_mat, cv::COLOR_BGR2YUV_I420);

  uint8_t *yuv = yuv_mat.ptr<uint8_t>();
  img_nv12 = cv::Mat(height * 3 / 2, width, CV_8UC1);
  uint8_t *ynv12 = img_nv12.ptr<uint8_t>();

  int32_t uv_height = height / 2;
  int32_t uv_width = width / 2;

  // copy y data
  int32_t y_size = height * width;
  memcpy(ynv12, yuv, y_size);

  // copy uv data
  uint8_t *nv12 = ynv12 + y_size;
  uint8_t *u_data = yuv + y_size;
  uint8_t *v_data = u_data + uv_height * uv_width;

  for (int32_t i = 0; i < uv_width * uv_height; i++) {
    *nv12++ = *u_data++;
    *nv12++ = *v_data++;
  }
  return 0;
}

float InferenceModule::quanti_shift(int32_t data, uint32_t shift) {
  return static_cast<float>(data) / static_cast<float>(1 << shift);
}

bool InferenceModule::is_quanti(hbDNNTensorProperties *property) {
  if (property->shift.shiftData) {
    return true;
  }
  return false;
}

void InferenceModule::get_bbox_result(hbDNNTensor *tensor,
                                      spJ5FrameMessage out_message) {
  hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  int *shape = tensor->properties.validShape.dimensionSize;
  int h_idx, w_idx, c_idx;
  get_hwc_index(tensor->properties.tensorLayout, &h_idx, &w_idx, &c_idx);
  int shape_w = shape[w_idx];
  int shape_c = shape[c_idx];

  for (auto i = 0; i < shape_w; i++) {
    float data[shape_c] = {0};
    for (auto j = 0; j < shape_c; j++) {
      float tmp;
      if (is_quanti(&tensor->properties)) {
        tmp = quanti_shift(reinterpret_cast<int32_t *>(
                               tensor->sysMem[0].virAddr)[i * shape_c + j],
                           tensor->properties.shift.shiftData[j]);
      } else {
        tmp = reinterpret_cast<float *>(
            tensor->sysMem[0].virAddr)[i * shape_c + j];
      }
      data[j] = tmp;
    }
    if (data[4] > 0) {
      auto box = std::make_shared<BBox>();
      box->x1_ = data[0];
      box->y1_ = data[1];
      box->x2_ = data[2];
      box->y2_ = data[3];
      box->score_ = data[4];
      if (detect_classify_.find(data[5]) != detect_classify_.end()) {
        box->specific_type_ = detect_classify_[data[5]];
      }
      LOGD << "box info: " << *box;
      auto target = std::make_shared<Target>();
      target->type_ = box->specific_type_;
      target->boxs_.push_back(box);
      out_message->targets_.push_back(target);
    }
  }
}

void InferenceModule::get_seg_result(hbDNNTensor *tensor,
                                     spJ5FrameMessage out_message) {
  hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  int *shape = tensor->properties.validShape.dimensionSize;

  int h_idx, w_idx, c_idx;
  get_hwc_index(tensor->properties.tensorLayout, &h_idx, &w_idx, &c_idx);

  int shape_w = shape[w_idx];
  int shape_h = shape[h_idx];
  int shape_c = shape[c_idx];

  auto seg = std::make_shared<Segmentation>();
  seg->width_ = shape_w;
  seg->height_ = shape_h;
  auto target = std::make_shared<Target>();
  target->segs_.push_back(seg);
  out_message->targets_.push_back(target);
  auto &one_data = seg->values_;
  int32_t *data = reinterpret_cast<int32_t *>(tensor->sysMem[0].virAddr);
  for (auto i = 0; i < shape_h; i++) {
    for (auto j = 0; j < shape_w; j++) {
      int offset = i * shape_w + j;
      int32_t c_max = -99999999;
      int index = 0;
      for (auto z = 0; z < shape_c; z++) {
        int32_t tmp = data[offset];
        if (tmp > c_max) {
          c_max = tmp;
          index = z;
        }
        offset += shape_h * shape_w;
      }
      if (index < 5) {  //  set to zero(RGB(0,0,0))
        one_data.push_back(0);
      } else {
        one_data.push_back(index);
      }
    }
  }
}

void InferenceModule::CoordinateTransform(spJ5FrameMessage out_message,
                                          int src_image_width,
                                          int src_image_height,
                                          int model_input_width,
                                          int model_input_height) {
  float ratio_w = static_cast<float>(src_image_width) / model_input_width;
  float ratio_h = static_cast<float>(src_image_height) / model_input_height;
  for (auto org_target : out_message->targets_) {
    for (auto org_box : org_target->boxs_) {
      org_box->x1_ *= ratio_w;
      org_box->y1_ *= ratio_h;
      org_box->x2_ *= ratio_w;
      org_box->y2_ *= ratio_h;
    }
  }
}
}  // namespace J5Sample
