/***************************************************************************
* COPYRIGHT NOTICE
* Copyright 2020 Horizon Robotics, Inc.
* All rights reserved.
***************************************************************************/
#ifndef __HB_VIN_DATA_INFO_H__
#define __HB_VIN_DATA_INFO_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef enum CAM_DATA_TYPE_S {
	HB_CAM_RAW_DATA = 0,
	HB_CAM_YUV_DATA = 1,
	HB_CAM_EMB_DATA = 2,
} CAM_DATA_TYPE_E;

typedef struct AWB_DATA {
	uint16_t WBG_R;
	uint16_t WBG_GR;
	uint16_t WBG_GB;
	uint16_t WBG_B;
}AWB_DATA_s;

typedef struct img_addr_info_s {
	uint16_t width;
	uint16_t height;
	uint16_t stride;
	uint64_t y_paddr;
	uint64_t c_paddr;
	uint64_t y_vaddr;
	uint64_t c_vaddr;
} img_addr_info_t;

typedef struct cam_img_info_s {
	int32_t g_id;
	int32_t slot_id;
	int32_t cam_id;
	int32_t frame_id;
	int64_t timestamp;
	img_addr_info_t img_addr;
} cam_img_info_t;

typedef struct {
	uint32_t frame_length;
	uint32_t line_length;
	uint32_t width;
	uint32_t height;
	float    fps;
	uint32_t pclk;
	uint32_t exp_num;
	uint32_t lines_per_second;
	char     version[10];
} sensor_parameter_t;

typedef struct {
	uint32_t id;
	uint16_t image_height;
	uint16_t image_width;
	uint16_t vendor;
	uint16_t version;
	uint8_t type;
	double 	focal_u;
	double 	focal_v;
	double 	center_u;
	double 	center_v;
	double 	hfov;
	double 	k1;
	double 	k2;
	double 	p1;
	double 	p2;
	double 	k3;
	double 	k4;
	double 	k5;
	double 	k6;
} sensor_intrinsic_parameter_t;

typedef struct {
	sensor_parameter_t sns_param;
	sensor_intrinsic_parameter_t sns_intrinsic_param;
} cam_parameter_t;

#ifdef __cplusplus
}
#endif

#endif
