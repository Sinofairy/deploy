/***************************************************************************
* COPYRIGHT NOTICE
* Copyright 2019 Horizon Robotics, Inc.
* All rights reserved.
***************************************************************************/
#ifndef J5_VIO_CIM_CIM_DMA_H_
#define J5_VIO_CIM_CIM_DMA_H_

#include "hb_buffer_mgr.h"
#include "hb_vin_data_info.h"
#include "hb_utils.h"

#define UV_PLANE_SHIFT 2u
#define UV_PLANE_CNT 2u
#define BUF_CONSECUTIVE
typedef struct cim_dma_data_desc {
	uint32_t    format;
	uint32_t    width;
	uint32_t    height;
	uint32_t    raw16_splict;
	uint32_t    dbg_clear;
	uint32_t    y_uv_swap;
} cim_dma_data_desc_t;

typedef struct cim_dma_func_desc {
	uint32_t    enable_dma;
	uint32_t    pack_mode;
	uint32_t    enable_id_decoder;
	uint32_t    enable_frame_id;
	uint32_t    enable_bypass;
	uint32_t    enable_pattern;
	uint32_t    hdr_mode;
	uint32_t    set_init_frame_id;
	uint32_t    set_bypass_channels;
	uint32_t	skip_frame;
	uint32_t	input_fps;
	uint32_t	output_fps;
} cim_dma_func_desc_t;

typedef struct cim_dma_input_mipi {
	cim_dma_data_desc_t   data;
	cim_dma_func_desc_t   func;
	uint32_t          enable;
	uint32_t          channels;
	uint32_t          rx_index;
	uint32_t          vc_index;
	uint32_t          tag_enable;
} cim_dma_input_mipi_t;

typedef struct cim_dma_input_iar {
	uint32_t          enable;
	cim_dma_func_desc_t   func;
} cim_dma_input_iar_t;

typedef struct cim_dma_input {
	cim_dma_input_mipi_t    mipi;
	cim_dma_input_iar_t     iar;
} cim_dma_input_t;

typedef struct cim_dma_output_ddr {
	uint32_t          enable;
	uint32_t          width;
	uint32_t          height;
	uint32_t          format;
	uint32_t          stride;
} cim_dma_output_ddr_t;

typedef struct cim_dma_output_mipi {
	uint32_t          enable;
	uint32_t          index;
} cim_dma_output_mipi_t;

typedef struct cim_dma_output {
	cim_dma_output_mipi_t    tx;
	cim_dma_output_ddr_t     ddr;
} cim_dma_output_t;

typedef struct cim_dma_cfg_s {
	cim_dma_input_t       input;
	cim_dma_output_t      output;
	uint32_t	buf_num;
	uint32_t	cached;
	uint32_t	use_ispbuf;
	uint32_t	buf_consecutive;
	uint32_t	water_level_mark;
	uint32_t	embeded_data_en;
	uint32_t	embeded_width;
	uint32_t	embeded_height;
	uint32_t 	reserved[5];
	uint32_t 	magicNumber;
} cim_dma_cfg_t;

typedef struct cim_dma_entity_s {
	uint32_t pipeline_id;
	uint8_t enable;
	int32_t fd_main;
	int32_t fd_bypass;
	cim_dma_cfg_t cim_dma_cfg;
	buffer_mgr_t *cim_dma_buf_mgr;
	buffer_mgr_t *emb_buf_mgr;
} cim_dma_entity_t;

cim_dma_entity_t *cim_dma_entity_create(uint32_t pipeline_id);
void cim_dma_entity_destroy(cim_dma_entity_t * handle);
int32_t cim_dma_entity_init(cim_dma_entity_t * handle);
int32_t cim_dma_entity_deinit(const cim_dma_entity_t * handle);
int32_t cim_dma_entity_parser_config(const cJSON *root, cim_dma_cfg_t * cfg);
int32_t cim_dma_entity_reset(const cim_dma_entity_t * handle);
int32_t cim_dma_entity_start(const cim_dma_entity_t * handle);
int32_t cim_dma_entity_stop(const cim_dma_entity_t * handle);
int32_t cim_dma_entity_put_buf(const cim_dma_entity_t * handle, CAM_DATA_TYPE_E data_type,
		const hb_vio_buffer_t * buf);
int32_t cim_dma_entity_get_buf(const cim_dma_entity_t * handle, CAM_DATA_TYPE_E data_type,
		hb_vio_buffer_t * buf);
int32_t cim_dma_entity_bypass_enable(cim_dma_entity_t * handle, uint32_t enable);
int32_t cim_dma_entity_set_fps_ctrl(const cim_dma_entity_t *handle,
		uint32_t skip_frame, uint32_t in_fps, uint32_t out_fps);

#endif //J5_VIO_CIM_CIM_DMA_H_
