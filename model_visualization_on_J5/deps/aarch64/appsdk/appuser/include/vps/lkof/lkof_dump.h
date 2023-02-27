/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2020 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/
#ifndef LKOF_DUMP_H
#define LKOF_DUMP_H
#include <stdint.h>
#include <stdio.h>
#include <lkof_device.h>
void debug_dump_set_paramters(struct lkof_settings_hw_s *settings_hw);
void debug_dump_src_dst_opt(src_sample_param_t	const *src_sample_param,
				dst_sample_param_t	const *dst_sample_param,
				dst_roi_t		const *dst_roi,
				opt_ctrl_t		const *opt_ctrl,
				block_parameter_t	const *block_parameter,
				result_addr_t			*result_addr,
				image_address_t     image_address);
#endif //LKOF_DUMP_H
