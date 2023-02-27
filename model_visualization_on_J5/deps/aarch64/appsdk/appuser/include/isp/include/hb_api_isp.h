/***************************************************************************
* COPYRIGHT NOTICE
* Copyright 2019 Horizon Robotics, Inc.
* All rights reserved.
***************************************************************************/
#ifndef __HB_API_ISP_H__
#define __HB_API_ISP_H__

#include "hb_comm_isp.h"

#ifdef __cplusplus
#if __cplusplus
extern "C"{
#endif
#endif /* __cplusplus */

extern int32_t hb_isp_run_algo(uint32_t pipeline_id);
extern int32_t hb_isp_pause_algo(uint32_t pipeline_id);
extern int32_t hb_isp_set_module_control(uint32_t pipeline_id, const isp_module_ctrl_u *mod_ctrl);
extern int32_t hb_isp_get_module_control(uint32_t pipeline_id, isp_module_ctrl_u *mod_ctrl);
extern int32_t hb_isp_get_ae_statistics(uint32_t pipeline_id, isp_statistics_t *ae_statistics, int32_t time_out);
extern int32_t hb_isp_release_ae_statistics(uint32_t pipeline_id, isp_statistics_t *ae_statistics);
extern int32_t hb_isp_get_awb_statistics(uint32_t pipeline_id, isp_statistics_t *awb_statistics, int32_t time_out);
extern int32_t hb_isp_release_awb_statistics(uint32_t pipeline_id, isp_statistics_t *awb_statistics);
extern int32_t hb_isp_set_log_level(uint32_t pipeline_id, const isp_log_level_e *log_level);
extern int32_t hb_isp_get_log_level(uint32_t pipeline_id, isp_log_level_e *log_level);
extern int32_t hb_isp_command(uint32_t pipeline_id, isp_cmd_api_t *cmd_api);
extern int32_t hb_isp_set_context(uint32_t pipeline_id, const isp_context_t *ptx);
extern int32_t hb_isp_get_context(uint32_t pipeline_id, isp_context_t *ptx);
extern int32_t hb_isp_set_ae_statistics_attr(uint32_t pipeline_id, const isp_ae_statistics_attr_t *ae_statistics_attr);
extern int32_t hb_isp_get_ae_statistics_attr(uint32_t pipeline_id, isp_ae_statistics_attr_t *ae_statistics_attr);
extern int32_t hb_isp_set_ae_info(uint32_t pipeline_id, const ae_info_t *ae_info);
extern int32_t hb_isp_get_ae_info(uint32_t pipeline_id, ae_info_t *ae_info);
extern int32_t hb_isp_set_awb_statistics_attr(uint32_t pipeline_id, const isp_awb_statistics_attr_t *awb_statistics_attr);
extern int32_t hb_isp_get_awb_statistics_attr(uint32_t pipeline_id, isp_awb_statistics_attr_t *awb_statistics_attr);
extern int32_t hb_isp_set_awb_info(uint32_t pipeline_id, const awb_info_t *awb_info);
extern int32_t hb_isp_get_awb_info(uint32_t pipeline_id, awb_info_t *awb_info);
extern int32_t hb_isp_get_version(uint32_t pipeline_id, char* isp_ver, char* algo_ver, char* calib_ver);
extern int32_t hb_isp_get_2a_info(uint32_t pipeline_id, isp_info_t *isp_info, int time_out);
extern int32_t hb_isp_dev_init(void);
extern int32_t hb_isp_dev_deinit(void);

extern int32_t hb_isp_set_calibration_param(uint32_t pipeline, const char *name, uint32_t param_type, uint32_t param_size, void *ptr);
extern int32_t hb_isp_get_calibration_param(uint32_t pipeline, const char *name, uint32_t param_type, uint32_t param_size, void *ptr);
extern int32_t hb_isp_get_command_range(uint32_t pipeline, uint32_t section, uint32_t command, uint32_t *max, uint32_t *min);
extern int32_t hb_isp_set_command_param(uint32_t pipeline, uint32_t section, uint32_t command, uint32_t data);
extern int32_t hb_isp_get_command_param(uint32_t pipeline, uint32_t section, uint32_t command, uint32_t *data);
extern int32_t hb_isp_get_hardware_range(uint32_t pipeline, const char *name, uint32_t *max, uint32_t *min);
extern int32_t hb_isp_set_hardware_param(uint32_t pipeline, const char *name, uint32_t param_size, uint32_t *ptr);
extern int32_t hb_isp_get_hardware_param(uint32_t pipeline, const char *name, uint32_t param_size, uint32_t *ptr);

/******************************************************************
 only used for engineer to debug isp status,
********************************************************************/
extern int32_t hb_isp_write_reg(uint32_t pipeline_id, const isp_reg_t *isp_reg);
/******************************************************************
 only used for engineer to debug isp status,
********************************************************************/
extern int32_t hb_isp_read_reg(uint32_t pipeline_id, isp_reg_t *isp_reg);

extern int32_t hb_isp_calib_lut(uint32_t pipeline_id, isp_calib_lut_t *calib_api);
extern int32_t hb_isp_lut_reg_rw(uint32_t pipeline_id, isp_lut_reg_t *isp_lut_reg);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */

#endif
