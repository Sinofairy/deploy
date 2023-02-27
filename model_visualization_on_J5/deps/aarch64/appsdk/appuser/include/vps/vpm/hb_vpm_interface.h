/***************************************************************************
* COPYRIGHT NOTICE
* Copyright 2019 Horizon Robotics, Inc.
* All rights reserved.
***************************************************************************/
#ifndef J5_VPM_INC_HB_VPM_INTERFACE_H
#define J5_VPM_INC_HB_VPM_INTERFACE_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "hb_vpm_data_info.h"

#define HB_ROTATION_0     0
#define HB_ROTATION_90    90
#define HB_ROTATION_180   180
#define HB_ROTATION_270   270

//---------------------------------------------------------
/**
 * parse the pipeline json configure according to cfg_file and init the pipeline
 * @param[in] cfg_file         pointer to json configure file
 * @return 0 if success
 */
int32_t hb_vpm_init(const char *cfg_file);
//---------------------------------------------------------
/**
 * init the pipeline by pipeline id and configure file
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] cfg_file         pointer to json configure file
 * @return 0 if success
 */
int32_t hb_vpm_init_pipeline(uint32_t pipeline_id, const char *cfg_file);
//---------------------------------------------------------
/**
 * deinit the pipeline , stop pipeline node and deinit their entity, then destroy the entity
 * @return 0 if success
 */
int32_t hb_vpm_deinit(void);
//---------------------------------------------------------
/**
 * deinit the pipeline by pipeline id , stop pipeline node and deinit their entity
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @return 0 if success
 */
int32_t hb_vpm_deinit_pipeline(uint32_t pipeline_id);
//---------------------------------------------------------
/**
 * start the pipeline by pipeline id, start streaming for ISP, PYM node
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @return 0 if success
 */
int32_t hb_vpm_start_pipeline(uint32_t pipeline_id);
//---------------------------------------------------------
/**
 * stop the pipeline by pipeline id, stop streaming for ISP, PYM node
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @return 0 if success
 */
int32_t hb_vpm_stop_pipeline(uint32_t pipeline_id);
//---------------------------------------------------------
/**
 * set vpm module callback func
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] type             callback data type
 * @param[in] cb               pointer to data callback function
 * @return 0 if success
 */
int32_t hb_vpm_set_callbacks(uint32_t pipeline_id, VIO_CALLBACK_TYPE_E type, data_cb cb);
//---------------------------------------------------------
/**
 * set msg information to pipeline
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] type             message information type
 * @param[in] info             pointer to message data
 * @return 0 if success
 */
int32_t hb_vpm_set_param(uint32_t pipeline_id, VIO_INFO_TYPE_E info_type, const void *info);
//---------------------------------------------------------
/**
 * get msg information from pipeline
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] type             message information type
 * @param[out] info            pointer to message data
 * @return 0 if success
 */
int32_t hb_vpm_get_param(uint32_t pipeline_id, VIO_INFO_TYPE_E info_type, void *info);
//---------------------------------------------------------
/**
 * get buffer from designated node of pipeline
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] data_type        data type ,represent which node and node data format
 * @param[out] data            pointer to data buffer
 * @return 0 if success
 */
int32_t hb_vpm_get_data(uint32_t pipeline_id, VIO_DATA_TYPE_E data_type, void *data);
//---------------------------------------------------------
/**
 * In the specified time zone, get buffer from designated node of pipeline 
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] data_type        data type ,represent which node and node data format
 * @param[out] data            pointer to data buffer
 * @param[in] cond_time        get the closest buffer in the past time zone
 * @return 0 if success
 */
int32_t hb_vpm_get_data_conditional(uint32_t pipeline_id, VIO_DATA_TYPE_E data_type,
				void *data, int32_t cond_time);
//---------------------------------------------------------
/**
 * free ipu node buffer
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] dst_img_info     pointer to buffer information
 * @return 0 if success
 */
int32_t hb_vpm_free_ipubuf(uint32_t pipeline_id, const hb_vio_buffer_t * dst_img_info);
//---------------------------------------------------------
/**
 * free isp node buffer
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] dst_img_info     pointer to buffer information
 * @return 0 if success
 */
int32_t hb_vpm_free_ispbuf(uint32_t pipeline_id, const hb_vio_buffer_t * dst_img_info);
//---------------------------------------------------------
/**
 * free pym node buffer
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] data_type        data type ,represent which node and node data format
 * @param[in] dst_img_info     pointer to buffer information
 * @return 0 if success
 */
int32_t hb_vpm_free_pymbuf(uint32_t pipeline_id, VIO_DATA_TYPE_E data_type, const void *img_info);
//---------------------------------------------------------
/**
 * queue pym src buffer and process
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] src_img_info     pointer to source buffer information
 * @return 0 if success
 */
int32_t hb_vpm_run_pym(uint32_t pipeline_id, const hb_vio_buffer_t *src_img_info);
//---------------------------------------------------------
/**
 * free gdc node buffer
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] dst_img_info     pointer to gdc destination buffer information
 * @return 0 if success
 */
int32_t hb_vpm_free_gdcbuf(uint32_t pipeline_id, const hb_vio_buffer_t *dst_img_info);
//---------------------------------------------------------
/**
 * process gdc node
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] src_img_info     pointer to gdc source buffer information
 * @param[in] dst_img_info     pointer to gdc destination buffer information
 * @param[in] rotate           gdc buffer process rotation parameter
 * @return 0 if success
 */
int32_t hb_vpm_run_gdc(uint32_t pipeline_id, const hb_vio_buffer_t *src_img_info,
					hb_vio_buffer_t * dst_img_info, int32_t rotate);
//---------------------------------------------------------
/**
 * free gdc configure buffer
 * @param[in] src_img_info     pointer to gdc configure buffer
 * @return 0 if success
 */
void hb_vpm_free_gdc_cfg(uint32_t* cfg_buf);
//---------------------------------------------------------
/**
 * init gdc and calculate gdc transformation
 * @param[in] gdc_param        pointer to gdc parameter ,including resoluti00n ,format etc
 * @param[in] windows     	   array of windows parameters
 * @param[in] wnd_num          number of windows
 * @param[in] cfg_buf          pointer to configure buffer pointer
 * @param[out] cfg_size        calculated configure buffer size
 * @return 0 if success
 */
int32_t hb_vpm_gen_gdc_cfg(const param_t *gdc_param, const window_t *windows,
									uint32_t wnd_num, void **cfg_buf, uint64_t *cfg_size);
//---------------------------------------------------------
/**
 * set gdc configure bin by configure buffer
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] cfg_buf          pointer to configure buffer
 * @param[in] cfg_size         configure buffer size
 * @return 0 if success
 */
int32_t   hb_vpm_set_gdc_cfg(uint32_t pipeline_id, const uint32_t *cfg_buf, uint64_t cfg_size);
//---------------------------------------------------------
/**
 * set gdc configure bin by configure buffer and check the gdc id is valid or not
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] gdc_id           gdc id,represent which gdc channel
 * @param[in] cfg_buf          pointer to configure buffer
 * @param[in] cfg_size         configure buffer size
 * @return 0 if success
 */
int32_t   hb_vpm_set_gdc_cfg_opt(uint32_t pipeline_id, uint32_t gdc_id,
							const uint32_t *cfg_buf, uint64_t cfg_size);//comp xj3
//---------------------------------------------------------
/**
 * check gdc id is valid ,if so, process gdc node
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] gdc_id           gdc id,represent which gdc channel
 * @param[in] src_img_info     pointer to gdc source buffer information
 * @param[in] dst_img_info     pointer to gdc destination buffer information
 * @param[in] rotate           gdc buffer process rotation parameter
 * @return 0 if success
 */
int32_t hb_vpm_run_gdc_opt(uint32_t pipeline_id,
						uint32_t gdc_id,
						const hb_vio_buffer_t * src_img_info,
						hb_vio_buffer_t * dst_img_info,
						int32_t rotate);//comp xj3
//---------------------------------------------------------
/**
 * check gdc id is valid ,if so, process gdc node and use user gdc configure
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] gdc_id           gdc id,represent which gdc channel
 * @param[in] gdc_cfg          pointer to gdc user configure
 * @param[in] src_img_info     pointer to gdc source buffer information
 * @param[in] dst_img_info     pointer to gdc destination buffer information
 * @param[in] rotate           gdc buffer process rotation parameter
 * @return 0 if success
 */
int32_t hb_vpm_run_gdc_adv(
						uint32_t pipeline_id,
						uint32_t gdc_id,
						const gdc_config_t *gdc_cfg,
						const hb_vio_buffer_t *src_img_info,
						hb_vio_buffer_t *dst_img_info,
						int32_t rotate);
//---------------------------------------------------------
/**
 * raw dump check
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] raw_img          pointer to raw buffer 
 * @param[in] gdc_cfg          pointer to yuv buffer
 * @return 0 if success
 */						
int32_t hb_vpm_raw_dump(uint32_t pipeline_id, const hb_vio_buffer_t * raw_img, const hb_vio_buffer_t * isp_yuv);
//---------------------------------------------------------
/**
 * raw dump check
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] feedback_src     pointer to feadback source buffer 
 * @param[in] gdc_cfg          pointer to yuv buffer
 * @return 0 if success
 */		
int32_t hb_vpm_raw_feedback(uint32_t pipeline_id, const hb_vio_buffer_t *feedback_src,
    const hb_vio_buffer_t *isp_dst_yuv);
//---------------------------------------------------------
/**
 * put raw buffer to isp and process
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] feedback_src     pointer to feadback source buffer 
 * @param[in] timeout          time out now is no use
 * @return 0 if success
 */	
int32_t hb_vpm_run_raw(uint32_t pipeline_id, const hb_vio_buffer_t *feedback_src, int32_t timeout);
//---------------------------------------------------------
/**
 * flush pym buffer to memory
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] pym_buf     	   pointer to pym buffer
 * @return 0 if success
 */	
int32_t hb_vpm_pym_flush_debug(uint32_t pipeline_id, const pym_buffer_v2_t *pym_buf);
//---------------------------------------------------------
/**
 * invalid pym buffer cache
 * @param[in] pipeline_id      pipeline id, represent which pipeline
 * @param[in] pym_buf          pointer to pym buffer 
 * @return 0 if success
 */	
int32_t hb_vpm_pym_invalid_cache_debug(uint32_t pipeline_id, const pym_buffer_v2_t *pym_buf);
//---------------------------------------------------------
#ifdef __cplusplus
}
#endif
//---------------------------------------------------------
#endif // J5_VPM_INC_HB_VPM_INTERFACE_H_
