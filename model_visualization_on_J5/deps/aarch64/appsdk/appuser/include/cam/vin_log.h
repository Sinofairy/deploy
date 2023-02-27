/***************************************************************************
* COPYRIGHT NOTICE
* Copyright 2019 Horizon Robotics, Inc.
* All rights reserved.
***************************************************************************/
#ifndef J5_VIN_UTILS_HB_UTILS_H_
#define J5_VIN_UTILS_HB_UTILS_H_

#include "logging.h"

#define MAX_VIN_PRINT_LENGTH 512
#define MAX_VIN_LARGE_PRINT_LENGTH 1024
#define HB_TIME_CALCULATE_UNIT_1000      1000
#define HB_TIME_CALCULATE_UNIT_1000000   1000000
#define HB_TIME_CALCULATE_UNIT_FLOAT_1000000   1000000.0f

#define J5_VIN_DEBUG 1

#ifndef pr_fmt
#define pr_fmt(fmt) fmt
#endif

typedef enum vin_log_level {
	VIN_NONE = 0,
	VIN_ERR,
	VIN_WARN,
	VIN_INFO,
	VIN_DEBUG,
	MAX_VIN_LOG_LEVEL
} vin_log_level_e;

#if J5_VIN_DEBUG
#define vin_err(fmt, ...)		hb_vin_log_warpper(VIN_ERR, pr_fmt(fmt), ##__VA_ARGS__)  /*PRQA S 0342,1038*/
#define vin_warn(fmt, ...)		hb_vin_log_warpper(VIN_WARN, pr_fmt(fmt), ##__VA_ARGS__)  /*PRQA S 0342,1038*/
#define vin_dbg(fmt, ...)		hb_vin_log_warpper(VIN_DEBUG, pr_fmt(fmt), ##__VA_ARGS__)  /*PRQA S 0342,1038*/
#define vin_info(fmt, ...)		hb_vin_log_warpper(VIN_INFO, pr_fmt(fmt), ##__VA_ARGS__)  /*PRQA S 0342,1038*/
#else
#define vin_err(fmt, ...)		hb_utils_log_warpper(VIN_ERR, pr_fmt(fmt), ##__VA_ARGS__)  /*PRQA S 0342,1038*/
#endif

#define android_printLog_dbg(fmt, ...)    android_printLog(3, NULL, fmt, ##__VA_ARGS__)  /*PRQA S 0342,1038*/
#define android_printLog_info(fmt, ...)    android_printLog(4, NULL, fmt, ##__VA_ARGS__)  /*PRQA S 0342,1038*/
#define android_printLog_warn(fmt, ...)    android_printLog(5, NULL, fmt, ##__VA_ARGS__)  /*PRQA S 0342,1038*/
#define android_printLog_err(fmt, ...)    android_printLog(6, NULL, fmt, ##__VA_ARGS__)  /*PRQA S 0342,1038*/

void hb_vin_log_warpper(vin_log_level_e level, const char *format, ...);//no need return value for check
#endif //J5_VIN_UTILS_HB_UTILS_H_

