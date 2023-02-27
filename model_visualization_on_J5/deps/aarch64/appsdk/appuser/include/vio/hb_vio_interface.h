/*
 * Horizon Robotics
 *
 * Copyright (C) 2020 Horizon Robotics Inc.
 * All rights reserved.
 * Author: 
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef HB_VIO_INTERFACE_H__
#define HB_VIO_INTERFACE_H__

#ifdef __cplusplus
extern "C"
{
#endif

#include <string.h>
#include <stdint.h>
#include "hb_vin_data_info.h"
#include "hb_vpm_data_info.h"

#define HB_VIO_ISP_DUMP_EN
#define HB_VIO_IPU_EN
#define MULTI_FLAG_PATH "/sys/module/hobot_cim/parameters/vio_mp_en"

#define OPEN_FLAG_RO        (0)
#define OPEN_FLAG_RW        (2)
#define HB_ISP_CTX_ID_0     (0)
#define HB_ISP_CTX_ID_1     (1)
#define HB_ISP_CTX_ID_2     (2)
#define HB_ISP_CTX_ID_3     (3)
#define HB_ISP_SW_ID_OFFSET (4)

// vio state check
#define HB_DEFAULT          (0)
#define HB_VIN_INIT         (1)
#define HB_VIN_DEINIT       (2)
#define HB_VIN_START        (3)
#define HB_VIN_STOP         (4)
#define HB_VPM_INIT         (5)
#define HB_VPM_DEINIT       (6)
#define HB_VPM_START        (7)
#define HB_VPM_STOP         (8)

/********************************* VIN API *********************************/
/**
 * hb_cam_init : init cam, include cmos, mipi, cim/cimdma.
 * @cfg_index: config index, which cam you want to choose;
 * @cfg_file: config file path, include cmos/mipi/cim/cimdma configs;
 */
int32_t hb_cam_init(uint32_t cfg_index, const char *cfg_file);

/**
 * hb_cam_deinit : deinit cam, include cmos, mipi, cim/cimdma.
 * @cfg_index: config index, which cam you want to choose;
 */
int32_t hb_cam_deinit(uint32_t cfg_index);

/**
 * hb_cam_start : start cam, include cmos, mipi, cim/cimdma.
 * @port: camera port, which cam you want to choose;
 */
int32_t hb_cam_start(uint32_t port);

/**
 * hb_cam_stop : deinit cam, include cmos, mipi, cim/cimdma.
 * @port: camera port, which cam you want to choose;
 */
int32_t hb_cam_stop(uint32_t port);

/**
 * hb_cam_start_all : start all cam which you init(in config index), include cmos, mipi, cim/cimdma.
 */
int32_t hb_cam_start_all(void);

/**
 * hb_cam_stop_all : stop all cam which you init(in config index), include cmos, mipi, cim/cimdma.
 */
int32_t hb_cam_stop_all(void);

/**
 * hb_cam_reset : reset cam, include cmos, mipi, cim/cimdma.
 * @port: camera port, which cam you want to choose;
 */
int32_t hb_cam_reset(uint32_t port);

/**
 * hb_cam_power_on : power on cmos.
 * @port: cmos port, which cmos you want to choose;
 */
int32_t hb_cam_power_on(uint32_t port);

/**
 * hb_cam_power_off : power off cmos.
 * @port: cmos port, which cmos you want to choose;
 */
int32_t hb_cam_power_off(uint32_t port);

/**
 * hb_cam_get_fps : deinit cam, include cmos, mipi, cim/cimdma.
 * @port: cam port, which cam you want to choose;
 * @fps: output parameters, fps == what you set in config;
 */
int32_t hb_cam_get_fps(uint32_t port, uint32_t *fps);

/**
 * hb_cam_get_img : get cam img info(only support in xj2).
 * @cam_img_info: output parameters, inlcude vaddr/paddr/w/h etc;
 */
int32_t hb_cam_get_img(cam_img_info_t *cam_img_info);

/**
 * hb_cam_free_img : free cam info(only support in xj2).
 * @cam_img_info: output parameters, inlcude vaddr/paddr/w/h etc.
 */
int32_t hb_cam_free_img(cam_img_info_t *cam_img_info);

/**
 * hb_cam_clean_img : clean cam info(only support in xj2).
 * @cam_img_info: output parameters, inlcude vaddr/paddr/w/h etc.
 */
int32_t hb_cam_clean_img(cam_img_info_t *cam_img_info);

/**
 * hb_cam_get_data : get cam data info.
 * @port: cam port, which cam you want to choose;
 * @data_type: data type, raw/yuv etc.
 * @data: cam image(from cim/cimdma)
 */
int32_t hb_cam_get_data(uint32_t port, CAM_DATA_TYPE_E data_type, void* data);

/**
 * hb_cam_free_data : free cam data info.
 * @port: cam port, which cam you want to choose;
 * @data_type: data type, raw/yuv etc.
 * @data: cam image(from cim/cimdma)
 */
int32_t hb_cam_free_data(uint32_t port, CAM_DATA_TYPE_E data_type, void* data);

/**
 * hb_cam_bypass_enable : deinit cam, include cmos, mipi, cim/cimdma.
 * @cfg_index:
 */
int32_t hb_cam_bypass_enable(uint32_t port, int32_t enable);

/**
 * hb_cam_set_fps_ctrl : skip frame in cim/cimdma side dynamically.
 * @port: cam port: which cam port you want to choose;
 * @skip_frame: 1 enable; 0 disable;
 * @in_fps: input fps
 * @out_fps: output fps
 */
int32_t hb_cam_set_fps_ctrl(uint32_t port, uint32_t skip_frame, uint32_t in_fps, uint32_t out_fps);

/**
 * hb_cam_dynamic_switch_fps : switch coms setting.
 * @port: cmos port, which cmos you want to choose;
 * @fps: switch cmos fps setting;
 */
int32_t hb_cam_dynamic_switch_fps(uint32_t port, uint32_t fps);

/**
 * hb_cam_dynamic_switch_mode : switch coms setting.
 * @port: cmos port, which cam you want to choose;
 * @mode: switch cmos mode setting;
 */
int32_t hb_cam_dynamic_switch_mode(uint32_t port, uint32_t mode);

/**
 * hb_cam_dynamic_switch : switch coms setting.
 * @port: cam port, which cam you want to choose;
 * @fps: switch cmos fps setting;
 * @resolution: switch cmos resolution setting;
 */
int32_t hb_cam_dynamic_switch(uint32_t port, uint32_t fps, uint32_t resolution);

/**
 * hb_cam_set_mclk : set sensor mclk(only xj3 support).
 * @entry_num: which mipi host you choose;
 * @mclk:  main clk;
 */
int32_t hb_cam_set_mclk(uint32_t entry_num, uint32_t mclk);

/**
 * hb_cam_enable_mclk : enable sensor mclk(only xj3 support).
 * @entry_num: which mipi host you choose;;
 */
int32_t hb_cam_enable_mclk(uint32_t entry_num);

/**
 * hb_cam_disable_mclk : disable sensor mclk(only xj3 support).
 * @entry_num: which mipi host you choose;;
 */
int32_t hb_cam_disable_mclk(uint32_t entry_num);

/**
 * hb_cam_extern_isp_reset :  reset independent external isp.
 * @port: cam port, which cam you want to choose;
 */
int32_t hb_cam_extern_isp_reset(uint32_t port);

/**
 * hb_cam_extern_isp_poweroff : poweroff independent external isp.
 * @port: cam port, which cam you want to choose;
 */
int32_t hb_cam_extern_isp_poweroff(uint32_t port);

/**
 * hb_cam_extern_isp_poweron : poweron independent external isp.
 * @port: cam port, which cam you want to choose;
 */
int32_t hb_cam_extern_isp_poweron(uint32_t port);


/********************************* for i2c/spi rw *********************************/
/**
 * hb_cam_i2c_read : for i2c read.
 * @port: cam port, which cam you want to choose;
 * @reg_addr: which addr you want to read;
 */
int32_t hb_cam_i2c_read(uint32_t port, uint32_t reg_addr);

/**
 * hb_cam_i2c_read_byte : for i2c read.
 * @port: cam port, which cam you want to choose;
 * @reg_addr: which addr you want to read;
 */
int32_t hb_cam_i2c_read_byte(uint32_t port, uint32_t reg_addr);

/**
 * hb_cam_i2c_write : for i2c write.
 * @port: cam port, which cam you want to choose;
 * @reg_addr: which addr you want to read;
 * @value: what you write to addr;
 */
int32_t hb_cam_i2c_write(uint32_t port, uint32_t reg_addr, uint16_t value);

/**
 * hb_cam_i2c_write_byte : for i2c write.
 * @port: cam port, which cam you want to choose;
 * @reg_addr: which addr you want to read;
 * @value: what you write to addr;
 */
int32_t hb_cam_i2c_write_byte(uint32_t port, uint32_t reg_addr, uint8_t value);

/**
 * hb_cam_i2c_block_write : for i2c write.
 * @port: cam port, which cam you want to choose;
 * @subdev: sub device;
 * @reg_addr: your block's first addr;
 * @buffer: save what you write;
 * @size: block size;
 */
int32_t hb_cam_i2c_block_write(uint32_t port, uint32_t subdev,
                                       uint32_t reg_addr, char *buffer, uint32_t size);

/**
 * hb_cam_i2c_block_read : for i2c read.
 * @port: cam port, which cam you want to choose;
 * @subdev: sub device;
 * @reg_addr: your block's first addr;
 * @buffer: save what you read;
 * @size: block size;
 */
int32_t hb_cam_i2c_block_read(uint32_t port, uint32_t subdev,
                                      uint32_t reg_addr, char *buffer, uint32_t size);

/**
 * hb_cam_spi_block_write : for spi write.
 * @port: cam port, which cam you want to choose;
 * @subdev: sub device;
 * @reg_addr: your block's first addr;
 * @buffer: save what you write;
 * @size: block size;
 */
int32_t hb_cam_spi_block_write(uint32_t port, uint32_t subdev,
                                        uint32_t reg_addr, char *buffer, uint32_t size);

/**
 * hb_cam_spi_block_read : for spi read.
 * @port: cam port, which cam you want to choose;
 * @subdev: sub device;
 * @reg_addr: your block's first addr;
 * @buffer: save what you read;
 * @size: block size;
 */
int32_t hb_cam_spi_block_read(uint32_t port, uint32_t subdev,
                                       uint32_t reg_addr, char *buffer, uint32_t size);

/**
 * hb_cam_ipi_reset : enable/disable ipi.
 * @entry_num: mipi host num;
 * @ipi_index: mipi host ipi index;
 * @enable: enable/disable;
 */
int32_t hb_cam_ipi_reset(uint32_t entry_num, uint32_t ipi_index, uint32_t enable);
/************************************ VPM API ************************************/
/**
 * hb_vio_init : for vio init, include pym/isp/gdc.
 * @cfg_file: vio init config setting;
 */
int32_t hb_vio_init(const char *cfg_file);

/**
 * hb_vio_deinit : for vio deinit, include pym/isp/gdc.
 */
int32_t hb_vio_deinit(void);

/**
 * hb_vio_start_pipeline : choose a vio pipeline start, include pym/isp/gdc.
 * @pipeline_id: which vio pipeline you want to start;
 */
int32_t hb_vio_start_pipeline(uint32_t pipeline_id);

/**
 * hb_vio_stop_pipeline : choose a vio pipeline stop, include pym/isp/gdc.
 * @pipeline_id: which vio pipeline you want to stop;
 */
int32_t hb_vio_stop_pipeline(uint32_t pipeline_id);

/**
 * hb_vio_set_callbacks : have been removed, not suport now;
 */
int32_t hb_vio_set_callbacks(uint32_t pipeline_id, VIO_CALLBACK_TYPE_E type, data_cb cb);

/**
 * hb_vio_set_param : set vio paramters.
 * @pipeline_id: which vio pipeline you want to set;
 * @info_type: which type you want to set;
 * @info: what you want to set;
 */
int32_t hb_vio_set_param(uint32_t pipeline_id, VIO_INFO_TYPE_E info_type, void *info);

/**
 * hb_vio_get_param : get vio paramters.
 * @pipeline_id: which vio pipeline you want to set;
 * @info_type: which type you want to set;
 * @info: what you want to get;
 */
int32_t hb_vio_get_param(uint32_t pipeline_id, VIO_INFO_TYPE_E info_type, void *info);

/**
 * hb_vio_get_data : get vio info, such as pym/isp/gdc etc.
 * @pipeline_id: which vio pipeline info you want get;
 * @data_type: which type info you want to get;
 * @data: what you want to get;
 */
int32_t hb_vio_get_data(uint32_t pipeline_id, VIO_DATA_TYPE_E data_type, void *data);

/**
 * hb_vio_get_data_conditional : get vio info, such as pym/isp/gdc etc.
 * @pipeline_id: which vio pipeline info you want get;
 * @data_type: which type info you want to get;
 * @data: what you want to get;
 * @times: set whether the fetch data frame is a previous frame or a subsequent frame;
 */
int32_t hb_vio_get_data_conditional(uint32_t pipeline_id, VIO_DATA_TYPE_E data_type,
                                               void *data, int32_t times);

/**
 * hb_vio_run_pym : set src addr and enable pym.
  * @pipeline_id: which vio pipeline info you want run;
  * @src_img_info: src img info;
 */
int32_t hb_vio_run_pym(uint32_t pipeline_id, hb_vio_buffer_t * src_img_info);

#ifdef HB_VIO_IPU_EN
/**
 * hb_vio_free_ipubuf : free ipu info(only support in xj23).
 * @pipeline_id: which vio pipeline info you want free;
 * @dst_img_info: free ipu info;
 */
int32_t hb_vio_free_ipubuf(uint32_t pipeline_id, hb_vio_buffer_t * dst_img_info); //remove
#endif

/**
 * hb_vio_free_ispbuf : free isp info.
 * @pipeline_id: which vio pipeline info you want free;
 * @dst_img_info: free isp info;
 */
int32_t hb_vio_free_ispbuf(uint32_t pipeline_id, hb_vio_buffer_t * dst_img_info);

/**
 * hb_vio_free_pymbuf : free pym info.
 * @pipeline_id: which vio pipeline info you want free;
 * @data_type: which type you want to free;
 * @img_info: what you want to free;
 */
int32_t hb_vio_free_pymbuf(uint32_t pipeline_id, VIO_DATA_TYPE_E data_type, void *img_info);

/************************* for gdc api *************************/
/**
 * hb_vio_gen_gdc_cfg : get gdc config bin.
 * @gdc_parm: gdc cfg bin config;
 * @wnds: windows size;
 * @wnd_num: windows nums;
 * @cfg_buf: gdc config bin buff;
 * @cfg_size: gdc config bin buff size;
 */
int32_t hb_vio_gen_gdc_cfg(param_t *gdc_parm, window_t *wnds,
                                  uint32_t wnd_num, void **cfg_buf, uint64_t *cfg_size);

/**
 * hb_vio_set_gdc_cfg : set gdc cfg.
 * @pipeline_id: which vio pipeline info you want set;
 * @cfg_buf: gdc config bin buff;
 * @cfg_size: gdc config bin size;
 */
int32_t hb_vio_set_gdc_cfg(uint32_t pipeline_id, uint32_t* cfg_buf,
                                  uint64_t cfg_size);

/**
 * hb_vio_set_gdc_cfg_opt : switch gdc and set gdc cfg.
 * @pipeline_id: which vio pipeline info you want set;
 * @gdc_id: switch gdc hw id;
 * @cfg_buf: gdc config bin buff;
 * @cfg_size: gdc config bin size;
 */
int32_t hb_vio_set_gdc_cfg_opt(uint32_t pipeline_id, uint32_t gdc_id,
                                       uint32_t* cfg_buf, uint64_t cfg_size); //comp xj3

/**
 * hb_vio_free_gdc_cfg : free gdc config bin.
 * @cfg_buf: gdc config bin buff;
 */
void hb_vio_free_gdc_cfg(uint32_t* cfg_buf);

/**
 * hb_vio_run_gdc : run gdc.
 * @pipeline_id: which vio pipeline info you want do process;
 * @src_img_info: gdc source img info;
 * @dst_img_info: gdc output info;
 * @rotate: rotation;
 */
int32_t hb_vio_run_gdc(uint32_t pipeline_id, hb_vio_buffer_t * src_img_info,
                            hb_vio_buffer_t * dst_img_info, int32_t rotate);

/**
 * hb_vio_run_gdc_opt : switch gdc and run gdc.
 * @pipeline_id: which vio pipeline info you want do process;
 * @gdc_id: switch gdc hw id;
 * @src_img_info: gdc source img info;
 * @dst_img_info: gdc output info;
 * @rotate: rotation;
 */
int32_t hb_vio_run_gdc_opt(uint32_t pipeline_id,
                                  uint32_t gdc_id,
                                  hb_vio_buffer_t * src_img_info,
                                  hb_vio_buffer_t * dst_img_info,
                                  int32_t rotate);//comp xj3

/**
 * hb_vio_run_gdc_adv : switch gdc and set gdc bin and run gdc.
 * @pipeline_id: which vio pipeline info you want do process;
 * @gdc_id: switch gdc hw id;
 * @gdc_cfg: gdc config bin;
 * @src_img_info: gdc source img info;
 * @dst_img_info: gdc output info;
 * @rotate: rotation;
 */
int32_t hb_vio_run_gdc_adv(uint32_t pipeline_id,
                                  uint32_t gdc_id,
                                  const gdc_config_t *gdc_cfg,
                                  hb_vio_buffer_t *src_img_info,
                                  hb_vio_buffer_t *dst_img_info,
                                  int32_t rotate);

/**
 * hb_vio_free_gdcbuf : free gdc buff.
 * @pipeline_id: which vio pipeline info you want do free;
 * @dst_img_info: gdc output info;
 */
int32_t hb_vio_free_gdcbuf(uint32_t pipeline_id, hb_vio_buffer_t * dst_img_info);

/*********** for debug, may remove ***********/
#ifdef HB_VIO_ISP_DUMP_EN
/**
 * hb_vio_raw_dump : dump raw data(only xj3 support).
 * @pipeline_id: which vio pipeline info you want do dump;
 * @raw_img: dump raw img;
 * @yuv: dump yuv img;
 */
int32_t hb_vio_raw_dump(uint32_t pipeline_id, hb_vio_buffer_t * raw_img,
                              hb_vio_buffer_t * yuv);

/**
 * hb_vio_raw_feedback : sif feed back raw to isp(only xj3 support).
 * @pipeline_id: which vio pipeline info you want do feedback;
 * @feedback_src: feedback src;
 * @isp_dst_yuv: output yuv;
 */
int32_t hb_vio_raw_feedback(uint32_t pipeline_id, hb_vio_buffer_t * feedback_src,
                                    hb_vio_buffer_t * isp_dst_yuv);
#endif

/**
 * hb_vio_run_raw : run isp process raw data.
 * @pipeline_id: which vio pipeline info you want run;
 * @feedback_src: feedback src;
 * @timeout: process timeout;
 */
int32_t hb_vio_run_raw(uint32_t pipeline_id, hb_vio_buffer_t * feedback_src, int32_t timeout);

/**
 * hb_vio_cfg_check : check vin & vpm cfg.
 * @vpm_file: vio init config setting;
 * @vin_file: config file path, include cmos/mipi/cim/cimdma configs;
 * @cfg_index: config index, which cam you want to choose;
 */
int32_t hb_vio_cfg_check(const char* vpm_file, const char* vin_file, uint32_t cfg_index);

#ifdef __cplusplus
}
#endif

#endif // HB_VIO_INTERFACE_H__
