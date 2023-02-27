/***************************************************************************
* COPYRIGHT NOTICE
* Copyright 2019 Horizon Robotics, Inc.
* All rights reserved.
***************************************************************************/
#ifndef __HB_ISP_UTILS_H__
#define __HB_ISP_UTILS_H__ /*PRQA S ALL*/
#include "log.h"


#ifdef __cplusplus
#if __cplusplus
extern "C"{
#endif
#endif /* __cplusplus */

#define isp_err(fmt, ...)	ALOGE(fmt, ##__VA_ARGS__)
#define isp_warn(fmt, ...)	ALOGW(fmt, ##__VA_ARGS__)//pr_warn(fmt, ##__VA_ARGS__)
#define isp_dbg(fmt, ...)	ALOGD(fmt, ##__VA_ARGS__)//pr_debug(fmt, ##__VA_ARGS__)
#define isp_log(fmt, ...)	ALOGV(fmt, ##__VA_ARGS__)//pr_info(fmt, ##__VA_ARGS__)

#endif /* __HB_TYPE_H__ */
