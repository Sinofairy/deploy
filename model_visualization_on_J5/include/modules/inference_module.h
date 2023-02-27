/**
 * Copyright Horizon Robotics
 */

#ifndef _J5_SAMPLE_INCLUDE_MODULES_INFERENCE_MODULE_
#define _J5_SAMPLE_INCLUDE_MODULES_INFERENCE_MODULE_

#include <map>
#include <memory>
#include <string>

#include "base_message.h"
#include "dnn/hb_dnn.h"
#include "hobot/hobot.h"
#include "hobotlog/hobotlog.hpp"
#include "json_cfg_wrapper.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "thread_pool.h"

namespace J5Sample {
class InferenceModule : public hobot::Module {
 public:
  InferenceModule() : hobot::Module("J5Sample", "InferenceModule") {}
  explicit InferenceModule(std::string config_file)
      : hobot::Module("J5Sample", "InferenceModule"),
        config_file_(config_file) {}
  int Init(hobot::RunContext *context) override;

  void Reset() override;

  FORWARD_DECLARE(InferenceModule, 0);

 private:
  /**
   * load json config
   * @return: 0 if success, and -1 if failed
   */
  int LoadConfig();
  /**
   * do inference task
   * @param[in] pyramid_message: vio message input
   * @param[in] workflow: hobot workflow instance
   * @param[in] from: current module
   * @param[in] output_slot: output slot
   * @param[in] context: run context
   * @return: 0 if success, and -1 if failed
   */
  int DoProcess(spVioMessage pyramid_message, hobot::Workflow *workflow,
                Module *from, int output_slot, hobot::spRunContext context);
  /**
   * Prepare tensor and fill with yuv data
   * @param[out] input_tensor: tensor to be prepared
   * @param[in] yuv_data: the y data
   * @param[in] uv_data: the uv data
   * @param[in] image_height: image height
   * @param[in] image_width: image width
   * @return 0 if success otherwise -1
   */
  int prepare_nv12_tensor(hbDNNTensor *input_tensor, void *y_data,
                          void *uv_data, int image_height, int image_width);
  /**
   * Prepare tensor according to node info
   * @param[out] tensor: tensor to be prepared
   * @param[in] intput_count: input tensor count
   * @return 0 if success otherwise -1
   */
  int prepare_input_tensor(hbDNNTensor *tensor, int intput_count);
  /**
   * Prepare tensor according to node info
   * @param[out] tensor: tensor to be prepared
   * @param[in] output_count: output tensor count
   * @return 0 if success otherwise -1
   */
  int prepare_output_tensor(hbDNNTensor **tensor, int output_count);
  /**
   * Prepare tensor and fill with image data
   * @param[in] image_data
   * @param[out] tensor
   */
  void prepare_image_data(void *image_data, hbDNNTensor *tensor);
  /**
   * nhwc to nchw
   * @param[out] out_data0
   * @param[out] out_data1
   * @param[out] out_data2
   * @param[in] in_data
   * @param[in] height
   * @param[in] width
   * @return 0 if success otherwise -1
   */
  void nhwc_to_nchw(uint8_t *out_data0, uint8_t *out_data1, uint8_t *out_data2,
                    uint8_t *in_data, int height, int width);
  /**
   * Free image tensor
   * @param[in] tensor: Tensor to be released
   */
  void release_tensor(hbDNNTensor *tensor);
  /**
   * get box from output tensor
   * @param[in] tensor: Tensor data
   * @param[out] out_message: store smart message
   */
  void get_bbox_result(hbDNNTensor *tensor, spJ5FrameMessage out_message);
  /**
   * get segmentation from output tensor
   * @param[in] tensor: Tensor data
   * @param[out] out_message: store smart message
   */
  void get_seg_result(hbDNNTensor *tensor, spJ5FrameMessage out_message);
  /**
   * get hwc index
   * @param[in] layout: Tensor layout type,nchw or nhwc
   * @param[out] h_idx: store h index
   * @param[out] w_idx: store w index
   * @param[out] c_idx: store c index
   * @return 0 if success otherwise -1
   */
  int get_hwc_index(int32_t layout, int *h_idx, int *w_idx, int *c_idx);
  /**
   * padding image according to model input layout
   * @param[out] pOutputImg: store padded image data
   * @param[in] yuv_data: image y data
   * @param[in] uv_data: image uv data
   * @param[in] image_height: input image height
   * @param[in] image_width: input image width
   * @return 0 if success otherwise -1
   */
  int padding_nv12_image(uint8_t **pOutputImg, void *yuv_data, void *uv_data,
                         int image_height, int image_width);
  /**
   * check if the data needs to be transformed
   * @param[in] property: tensor property
   * @return ture if success otherwise false
   */
  inline bool is_quanti(hbDNNTensorProperties *property);
  /**
   * Transform the model output data
   * @param[in] data: data to transform
   * @param[in] shift: shift info
   * @return transformed data
   */
  inline float quanti_shift(int32_t data, uint32_t shift);
  /**
   * test create fake vio message
   * @return vio message
   */
  spVioMessage TestCreateVioMessage();
  /**
   * Read image and convert it to format NV12
   * @param[in] image_file: input image path
   * @param[in] yuv_height: yuv image height
   * @param[in] yuv_width: yuv image width
   * @param[out] img_nv12: nv12 image
   * @return: 0 if success, and -1 if failed
   */
  int32_t read_image_2_nv12(std::string &image_file, int yuv_height,
                            int yuv_width, cv::Mat &img_nv12);
  /**
   * Bgr image to nv12
   * @param[in] bgr_mat
   * @param[in] img_nv12
   * @return 0 if success otherwise -1
   */
  int32_t bgr_to_nv12(cv::Mat &bgr_mat, cv::Mat &img_nv12);
  /**
   * transform coordinate accroding to src img
   * @param[in out] out_message
   * @param[in] src_image_width
   * @param[in] src_image_height
   * @param[in] model_input_width
   * @param[in] model_input_hight
   */
  void CoordinateTransform(spJ5FrameMessage out_message, int src_image_width,
                           int src_image_height, int model_input_width,
                           int model_input_hight);

 private:
  bool is_inited_ = false;
  hbDNNHandle_t dnn_handle_ = 0;
  hbDNNTensorProperties input_properties_;
  hbPackedDNNHandle_t packed_dnn_handle_ = 0;
  int pym_level_ = 0;
  int model_intput_width_ = 0;
  int model_intput_height_ = 0;
  std::string config_file_ = "";
  std::string model_file_path_ = "";
  std::shared_ptr<JsonConfigWrapper> config_;
  std::map<int, std::string> detect_classify_;
  std::shared_ptr<CThreadPool> worker_ = nullptr;
};
}  // namespace J5Sample
#endif  // _J5_SAMPLE_INCLUDE_MODULES_INFERENCE_MODULE_
