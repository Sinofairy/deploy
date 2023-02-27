/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2020 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/
#ifndef LKOF_DEVICE_H
#define LKOF_DEVICE_H
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <lkof_instlist.h>
#include <hb_lkof_interface.h>

#define LKOF_DEVICE_NAME "/dev/lkof"
#define LKOF_LOG_PRINT 0

typedef struct pre_pym_layers_info_s {
	uint64_t	pre_layer0_ddr_addr;
	uint32_t	pre_layer1_ddr_offset;
	uint32_t	pre_layer2_ddr_offset;
	uint32_t	pre_layer3_ddr_offset;
	uint32_t	pre_layer4_ddr_offset;
}pre_pym_layers_info_t;

typedef struct next_pym_layers_info_s {
	uint64_t	next_layer0_ddr_addr;
	uint32_t	next_layer1_ddr_offset;
	uint32_t	next_layer2_ddr_offset;
	uint32_t	next_layer3_ddr_offset;
	uint32_t	next_layer4_ddr_offset;
}next_pym_layers_info_t;

typedef struct pym_layers_width_s {
	uint32_t	layer0_w;
	uint32_t	layer1_w;
	uint32_t	layer2_w;
	uint32_t	layer3_w;
	uint32_t	layer4_w;
}pym_layers_width_t;

typedef struct pym_layers_height_s {
	uint32_t	layer0_h;
	uint32_t	layer1_h;
	uint32_t	layer2_h;
	uint32_t	layer3_h;
	uint32_t	layer4_h;
}pym_layers_height_t;

typedef struct pym_layers_stride_s {
	uint32_t	layer0_stride;
	uint32_t	layer1_stride;
	uint32_t	layer2_stride;
	uint32_t	layer3_stride;
	uint32_t	layer4_stride;
}pym_layers_stride_t;

typedef struct total_addr_s {
	instlist_addr_t		instlist_addr;
	image_address_t		image_address;
	result_addr_t		*result_addr;
	fc_addr_t		fc_addr;
}total_addr_t;

typedef struct lkof_settings_hw_s {
	uint64_t		frame_id;
	int32_t			mode;
	int32_t 		user_error;
	uint32_t		magic_update;
	uint32_t		magic_num0;
	uint32_t		magic_num1;
	uint32_t		magic_num2;
	uint32_t		magic_num3;
	uint32_t		magic_num4;
	uint16_t		roi_num_in;
	uint64_t		funcall_ddr_addr;
	uint64_t		ins_ddr_addr;
	pre_pym_layers_info_t	pre_pym_layers_info;
	next_pym_layers_info_t	next_pym_layers_info;
	pym_layers_width_t	pym_layers_width;
	pym_layers_height_t	pym_layers_height;
	pym_layers_stride_t	pym_layers_stride;
	total_addr_t		total_addr;
}lkof_settings_hw_t;

int32_t lkof_device_open(void);
int32_t lkof_set_paramter(int32_t fd, struct lkof_settings_hw_s *setting_hw);
int32_t lkof_get_result(int32_t fd, struct lkof_settings_hw_s *setting_hw);
int32_t lkof_set_result_addr(int32_t fd, struct result_addr_s *result_addr);
int32_t lkof_set_roi_num(int32_t fd, uint16_t *roi_num);
int32_t lkof_get_kfifo_status(int32_t fd, struct fifo_info_status_s *fifo_info_status);
int32_t lkof_device_close(int32_t fd);

#define LKOF_IOCTL_MAGIC 'O'

#define LKOF_IOCTL_SET_HW_PARAMS                      _IOWR(LKOF_IOCTL_MAGIC, 0, struct lkof_settings_hw_s *)
#define LKOF_IOCTL_GET_RESULT                         _IOWR(LKOF_IOCTL_MAGIC, 1, struct lkof_settings_hw_s *)
#define LKOF_IOCTL_SET_RESULT_ADDR                    _IOWR(LKOF_IOCTL_MAGIC, 2, struct result_addr_s *)
#define LKOF_IOCTL_SET_ROI_NUM                        _IOWR(LKOF_IOCTL_MAGIC, 3, uint16_t *)
#define LKOF_IOCTL_GET_KFIFO_STATUS             _IOWR(LKOF_IOCTL_MAGIC, 4, struct fifo_info_status_s *)

#endif //LKOF_DEVICE_H
