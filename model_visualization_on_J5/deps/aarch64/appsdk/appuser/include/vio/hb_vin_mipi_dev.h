/***************************************************************************
* COPYRIGHT NOTICE
* Copyright 2016 Horizon Robotics, Inc.
* All rights reserved.
***************************************************************************/
/**
 * @file     mipi_dev.h
 * @brief    MIPI DEV Common define
 * @author   tarryzhang (tianyu.zhang@hobot.cc)
 * @date     2017/7/6
 * @version  V1.0
 * @par      Horizon Robotics
 */
#ifndef INC_HB_VIN_MIPI_DEV_H_
#define INC_HB_VIN_MIPI_DEV_H_

#ifdef __cplusplus
extern "C"
{
#endif


#include <linux/types.h>
#include "hb_vin_common.h"

int32_t hb_vin_mipi_dev_start(const entry_t *e);
int32_t hb_vin_mipi_dev_stop(const entry_t *e);
int32_t hb_vin_mipi_dev_init(entry_t *e);
int32_t hb_vin_mipi_dev_deinit(entry_t *e);
int32_t hb_vin_mipi_dev_parser_config(const void *root, entry_t *e);
int32_t hb_vin_mipi_dev_ipi_fatal(entry_t *e, int32_t ipi);
int32_t hb_vin_mipi_dev_ipi_get_info(entry_t *e, int32_t ipi, mipi_dev_ipi_info_t *info);
int32_t hb_vin_mipi_dev_ipi_set_info(entry_t *e, int32_t ipi, const mipi_dev_ipi_info_t *info);

#ifdef __cplusplus
}
#endif

#endif  // INC_HB_VIN_MIPI_DEV_H_
