/***************************************************************************
* COPYRIGHT NOTICE
* Copyright 2020 Horizon Robotics, Inc.
* All rights reserved.
***************************************************************************/
#ifndef VIDEOSTITCH_H
#define VIDEOSTITCH_H

#include <stdint.h>
#include <pthread.h>
#include "hb_videostitch_interface.h"

#define PLANE_NUM (2u)
#define LUT_NUM (2u)

#define VIDEOSTITCH_DEVICE_NAME "/dev/videostitch"

#define phys_addr_t uint64_t

#define FLAG0 (247u)
#define FLAG1 (40u)
#define FLAG2 (248u)
#define SRC_GAIN (256u)

typedef struct config_roi_s {
	union {
			uint32_t word0;
			struct {
				uint32_t direct:2;
				uint32_t roi_h:13;
				uint32_t roi_w:13;
				uint32_t UV_en:1;
				uint32_t mode:3;
			}Bits_word0;
	}config_word0;

	union {
			uint32_t word1;
			struct {
				uint32_t Y_Src0_addr;
			}Bits_word1;
	}config_word1;

	union {
			uint32_t word2;
			struct {
				uint32_t Y_Src1_addr;
			}Bits_word2;
	}config_word2;

	union {
			uint32_t word3;
			struct {
				uint32_t UV_Src0_addr;
			}Bits_word3;
	}config_word3;

	union {
			uint32_t word4;
			struct {
				uint32_t UV_Src1_addr;
			}Bits_word4;
	}config_word4;

	union {
			uint32_t word5;
			struct {
				uint32_t Y_Dst_addr;
			}Bits_word5;
	}config_word5;

	union {
			uint32_t word6;
			struct {
				uint32_t UV_Dst_addr;
			}Bits_word6;
	}config_word6;

	union {
			uint32_t word7;
			struct {
				uint32_t Alpha_addr;
			}Bits_word7;
	}config_word7;

	union {
			uint32_t word8;
			struct {
				uint32_t Beta_addr;
			}Bits_word8;
	}config_word8;

	union {
			uint32_t word9;
			struct {
				uint32_t Y_Src1_stride:16;
				uint32_t Y_Src0_stride:16;
			}Bits_word9;
	}config_word9;

	union {
			uint32_t word10;
			struct {
				uint32_t UV_Src1_stride:16;
				uint32_t UV_Src0_stride:16;
			}Bits_word10;
	}config_word10;

	union {
			uint32_t word11;
			struct {
				uint32_t UV_Dst_stride:16;
				uint32_t Y_Dst_stride:16;
			}Bits_word11;
	}config_word11;

	union {
			uint32_t word12;
			struct {
				uint32_t margin_inv:16;
				uint32_t margin:16;
			}Bits_word12;
	}config_word12;

	union {
			uint32_t word13;
			struct {
				uint32_t config_flag0:8;
				uint32_t reserve1:2;
				uint32_t src1_gain_y:11;
	            		uint32_t src0_gain_y:11;
			}Bits_word13;
	}config_word13;

	union {
			uint32_t word14;
			struct {
				uint32_t config_flag1:8;
				uint32_t reserve2:2;
				uint32_t src1_gain_u:11;
				uint32_t src0_gain_u:11;
			}Bits_word14;
	}config_word14;

	union {
			uint32_t word15;
			struct {
				uint32_t config_flag2:8;
				uint32_t reserve3:2;
				uint32_t src1_gain_v:11;
				uint32_t src0_gain_v:11;
			}Bits_word15;
	}config_word15;
}config_roi_t;

typedef struct videostitch_iommu_map_s {
	phys_addr_t        src_phy;
	phys_addr_t        map_phy;
	uint64_t           size;
}videostitch_iommu_map_t;

typedef struct videostitch_settings_lut_addr_info_s {
	phys_addr_t            lut_addr;
	uint32_t               lut_size;
}videostitch_settings_lut_addr_info_t;

typedef struct videostitch_settings_src_addr_info_s {
	phys_addr_t            src_addr;
	uint32_t               src_size;
}videostitch_settings_src_addr_info_t;

typedef struct videostitch_settings_dst_addr_info_s {
	phys_addr_t            dst_addr;
	uint32_t               dst_size;
}videostitch_settings_dst_addr_info_t;

typedef struct videostitch_settings_config_addr_info_s {
	phys_addr_t            unmap_config_addr;
	phys_addr_t            config_addr;
	uint32_t               config_size;
	uint32_t                  roi_num;
}videostitch_settings_config_addr_info_t;

typedef struct videostitch_address_info {
	phys_addr_t start;
	uint32_t  size;
}videostitch_address_info_t;

typedef struct videostitch_settings_s {
	videostitch_settings_lut_addr_info_t              lut_addr_info;
	videostitch_settings_src_addr_info_t              src_addr_info;
	videostitch_settings_config_addr_info_t           config_addr_info;
	videostitch_settings_dst_addr_info_t              dst_addr_info;
}videostitch_settings_t;

typedef struct videostitch_ctx_s {
	pthread_mutex_t         stitch_mutex;
	int32_t                 fd;
	uint16_t                inited;
	uint16_t                diag_send_count;
} videostitch_ctx_t;

//function
int32_t videostitch_process(uint32_t		roi_num,
				const roi_cfg_t	*roi_cfg,
				const frameinfo_t	*frameinfo,
				const result_addr_t	*result_addr,
				const cgp_addr_t	*cgp_addr,
				const lut_table_t	*lut_table);

int32_t videostitch_calc_outbuf_size(uint32_t width, uint32_t height, buffer_offset_t *buf_offset);
int32_t videostitch_cgp_size(uint32_t roi_num, uint32_t *size);

int32_t videostitch_init(void);
int32_t videostitch_deinit(void);


void printf_runtime_roi_info(config_roi_t *tmpcft_ptr, const roi_cfg_t* cfg_roi, const result_addr_t *result_addr);

#endif //VIDEOSTITCH_H
