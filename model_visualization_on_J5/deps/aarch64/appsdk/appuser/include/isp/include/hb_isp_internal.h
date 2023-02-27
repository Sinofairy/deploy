/***************************************************************************
* COPYRIGHT NOTICE
* Copyright 2019 Horizon Robotics, Inc.
* All rights reserved.
***************************************************************************/
#ifndef __HB_ISP_INTERNAL_H__
#define __HB_ISP_INTERNAL_H__

#include "hb_comm_isp.h"

#ifdef __cplusplus
#if __cplusplus
extern "C"{
#endif
#endif /* __cplusplus */

extern int32_t hb_isp_open_isp_res(uint32_t pipeline_id, const isp_pub_attr_t *pub_attr);
extern int32_t hb_isp_set_pub_attr(uint32_t pipeline_id, const isp_pub_attr_t *pub_attr);
extern int32_t hb_isp_get_pub_attr(uint32_t pipeline_id, isp_pub_attr_t *pub_attr);
extern int32_t hb_isp_pipe_init(uint32_t pipeline_id);
extern int32_t hb_isp_pipe_exit(uint32_t pipeline_id);
extern int32_t hb_isp_pipe_start(uint32_t pipeline_id);
extern int32_t hb_isp_pipe_stop(uint32_t pipeline_id);
extern int32_t hb_isp_set_output_attr(uint32_t pipeline_id, const isp_output_attr_t *output_attr);
extern int32_t hb_isp_get_output_attr(uint32_t pipeline_id, isp_output_attr_t *output_attr);
extern int32_t hb_isp_get_frame(uint32_t pipeline_id, isp_frame_info_t *frame_info, int32_t time_out);
extern int32_t hb_isp_release_frame(uint32_t pipeline_id, const isp_frame_info_t *frame_info);
extern int32_t hb_isp_set_input_frame(uint32_t pipeline_id, const isp_frame_info_t *input_frame);
extern int32_t hb_isp_get_input_frame(uint32_t pipeline_id, isp_frame_info_t *input_frame, int32_t time_out);
extern int32_t hb_isp_set_module_level(uint32_t pipeline_id, const isp_module_level_t *module_level);
extern  int32_t hb_isp_get_module_level(uint32_t pipeline_id, isp_module_level_t *module_level);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */

#endif
