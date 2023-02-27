/*
 * Horizon Robotics
 *
 *  Copyright (C) 2021 Horizon Robotics Inc.
 *  All rights reserved.
 *  Author: shaohui.cheng<shaohui.cheng@horizon.ai>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef J5_VIDEOSTITCH_INCLUDE_VIDEOSTITCH_SEND_SMFU_H_
#define J5_VIDEOSTITCH_INCLUDE_VIDEOSTITCH_SEND_SMFU_H_
#include <stdint.h>

#define EVENT_ERROR_LEVEL_1  (1u)
#define EVENT_ERROR_LEVEL_2  (2u)
#define EVENT_ERROR_LEVEL_3  (3u)
#define EVENT_LOW_8_BIT_MASK   (0x00ff)
#define EVENT_SHIFT_8_BIT   (8u)
#define EVENT_MODULE_ID   (34u)
#define EVENT_ENV_LEN   (4)
#define DIAG_ERR_STATUS_FAILED (3u)
#define STITCH_DIAG_MAX_COUNT (20u)

typedef enum stitch_hal_event_id_s {
    EventIdStitchHalInitDeinitErr = 60,
    EventIdStitchHalProcessParameterErr,
    EventIdStitchHalCalcOutbufParameterErr,
    EventIdStitchHalCgpSizeParameterErr,
    EventIdStitchHalCallDriverErr,
}stitch_hal_event_id_e;

enum sub_event_id {
	SUB_ID_0,
	SUB_ID_1,
	SUB_ID_2,
	SUB_ID_3,
	SUB_ID_4,
	SUB_ID_5,
	SUB_ID_6,
	SUB_ID_7,
	SUB_ID_8,
	SUB_ID_9,
	SUB_ID_10,
    SUB_ID_11,
	SUB_ID_12,
	SUB_ID_13,
	SUB_ID_14,
	SUB_ID_15,
	SUB_ID_16,
	SUB_ID_17,
	SUB_ID_18,
	SUB_ID_19,
	SUB_ID_20,
	SUB_ID_21,
	SUB_ID_22,
	SUB_ID_23,
	SUB_ID_24,
};



typedef enum call_driver_sub_event_id_s {
	SUBID_VIDEOSTITCH_HW_PROCESS_ERR = 1,
	SUBID_VIDEOSTITCH_HW_CHECK_ADDRESS_ERR,
	SUBID_VIDEOSTITCH_HW_MAP_ERR,
	SUBID_VIDEOSTITCH_HW_UNMAP_ERR,
	SUBID_VIDEOSTITCH_HW_FRAME_CHECK_ADDRESS_ERR,
	SUBID_VIDEOSTITCH_HW_RESULT_CHECK_ADDRESS_ERR,
	SUBID_VIDEOSTITCH_HW_LUT_CHECK_ADDRESS_ERR,
	SUBID_VIDEOSTITCH_HW_CGP_CHECK_ADDRESS_ERR,
	SUBID_VIDEOSTITCH_HW_FRAME_Y_MAP_ERR,
	SUBID_VIDEOSTITCH_HW_FRAME_UV_MAP_ERR,
	SUBID_VIDEOSTITCH_HW_RESULT_MAP_ERR,
	SUBID_VIDEOSTITCH_HW_LUT_ALPHA_MAP_ERR,
	SUBID_VIDEOSTITCH_HW_LUT_BETA_MAP_ERR,
	SUBID_VIDEOSTITCH_HW_CGP_MAP_ERR,
	SUBID_VIDEOSTITCH_HW_FRAME_Y_UNMAP_ERR,
	SUBID_VIDEOSTITCH_HW_FRAME_UV_UNMAP_ERR,
	SUBID_VIDEOSTITCH_HW_RESULT_UNMAP_ERR,
	SUBID_VIDEOSTITCH_HW_LUT_ALPHA_UNMAP_ERR,
	SUBID_VIDEOSTITCH_HW_LUT_BETA_UNMAP_ERR,
	SUBID_VIDEOSTITCH_HW_CGP_UNMAP_ERR,
} call_driver_sub_event_id_e;

void stitch_hal_send_sfmu_msg(stitch_hal_event_id_e event_id,
    uint8_t sub_event_id,
    uint32_t line_number,
    uint8_t err_level);

#endif //J5_VIDEOSTITCH_INCLUDE_VIDEOSTITCH_SEND_SMFU_H_


