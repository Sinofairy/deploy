/***************************************************************************
* COPYRIGHT NOTICE
* Copyright 2019 Horizon Robotics, Inc.
* All rights reserved.
***************************************************************************/
#ifndef J5_VIO_CIM_CIM_H_
#define J5_VIO_CIM_CIM_H_

#include "hb_buffer_mgr.h"
#include "hb_utils.h"
typedef struct cim_data_desc {
    uint32_t    format;
    uint32_t    width;
    uint32_t    height;
    uint32_t    raw16_splict;
    uint32_t    dbg_clear;
    uint32_t    y_uv_swap;
} cim_data_desc_t;

typedef struct cim_func_desc {
    uint32_t    enable_frame_id;
    uint32_t    enable_bypass;
    uint32_t    enable_pattern;
    uint32_t    hdr_mode;
    uint32_t    set_init_frame_id;
    uint32_t    set_bypass_channels;
	uint32_t	skip_frame;
	uint32_t	input_fps;
	uint32_t	output_fps;
} cim_func_desc_t;

typedef struct cim_input_dvp {
    cim_data_desc_t   data;
    cim_func_desc_t   func;
    uint32_t          enable;
	uint32_t          dcls_compare_en;//should remove fusa func was removed by niwei
	uint32_t          pr12_14bit;
	uint32_t          raw16_20_c;
	uint32_t          dvp_tag;//should remove
    uint32_t          hv_blank;
    uint32_t          tag_enable;
    uint32_t          data_mode; // for 16/20 bit
} cim_input_dvp_t;

typedef struct cim_input_mipi {
    cim_data_desc_t   data;
    cim_func_desc_t   func;
    uint32_t          enable;
    uint32_t          channels;
    uint32_t          rx_index;
    uint32_t          vc_index;
    uint32_t          tag_enable;
} cim_input_mipi_t;

typedef struct cim_input_iar {
    uint32_t          enable;
    cim_func_desc_t   func;
} cim_input_iar_t;

typedef struct cim_input {
    cim_input_dvp_t     dvp;
    cim_input_mipi_t    mipi;
    cim_input_iar_t     iar;
} cim_input_t;

typedef struct cim_output_isp {
    uint32_t          isp_ch0;
    uint32_t          isp_ch1;
    uint32_t          isp_ch2;
    uint32_t          isp_ch3;
} cim_output_isp_t;

typedef struct cim_output_pym {
    uint32_t          pym0_en;
	uint32_t          pym1_en;
} cim_output_pym_t;

typedef struct cim_output_mipi {
    uint32_t          enable;
    uint32_t          index;
} cim_output_mipi_t;

typedef struct cim_output {
    cim_output_mipi_t    tx[2];
    cim_output_pym_t     pym;
    cim_output_isp_t     isp[2];
} cim_output_t;

typedef struct cim_cfg_s {
    cim_input_t       input;
    cim_output_t      output;
	uint32_t reserved[10];
	uint32_t magicNumber;
} cim_cfg_t;

typedef struct cim_entity_s {
	uint32_t pipeline_id;
	uint8_t enable;
	int32_t fd_main;
	cim_cfg_t cim_cfg;
} cim_entity_t;

cim_entity_t *cim_entity_create(uint32_t pipeline_id);
void cim_entity_destroy(cim_entity_t *handle);
int32_t cim_entity_init(cim_entity_t *handle);
int32_t cim_entity_deinit(const cim_entity_t *handle);
int32_t cim_entity_parser_config(const cJSON *root, cim_cfg_t *cfg);
int32_t cim_entity_reset(const cim_entity_t *handle);
int32_t cim_entity_bypass_enable(const cim_entity_t *handle, uint32_t enable);
int32_t cim_entity_start(const cim_entity_t *handle);
int32_t cim_entity_stop(const cim_entity_t *handle);
int32_t cim_entity_mgr_flush(buffer_mgr_t *mgr);
int32_t cim_entity_set_fps_ctrl(const cim_entity_t *handle, uint32_t skip_frame, uint32_t in_fps, uint32_t out_fps);

#endif //J5_VIO_CIM_CIM_H_
