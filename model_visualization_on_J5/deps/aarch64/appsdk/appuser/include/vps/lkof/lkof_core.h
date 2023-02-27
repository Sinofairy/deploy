/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2020 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/
#ifndef LKOF_HAL_H
#define LKOF_HAL_H
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <lkof_device.h>
#include <hb_lkof_interface.h>
#include <lkof_instlist.h>
#include <param_fifo.h>

#define POLL_TIMEOUT (5000)

typedef struct env_var_s {
	int32_t need_point_check;
}env_var_t;

typedef struct lkof_ctx_s {
	int32_t			fd;
	instlist_ctx_t 		*instlist_ctx;
	pthread_t		thread_id;
	struct param_fifo 	*sync_running_params;
	struct param_fifo 	*async_running_params;
	struct param_fifo 	*sync_done_params;
	struct param_fifo 	*async_done_params;
	int32_t			running;
	int32_t			inited;
	struct env_var_s	env_var;
	pthread_mutex_t 	mutex;
}lkof_ctx_t;

lkof_ctx_t *lkof_hal_entity_init(void);

int32_t lkof_hal_entity_deinit (void);

lkof_ctx_t *lkof_create_ctx(void);

int32_t lkof_ftoi(float point, uint32_t *out);/*PRQA S ALL*/

int32_t lkof_itof(int32_t value, result_mem_type_t result_mem_type, float *out);/*PRQA S ALL*/

int32_t lkof_hal_trackpoint(image_address_t				image_address,
					uint32_t			roi_num_in,
					src_sample_param_t		const *src_sample_param,
					dst_sample_param_t		const *dst_sample_param,
					dst_roi_t			const *dst_roi,
					block_parameter_t		const *block_parameter,
					harris_t			const *harris,
					opt_ctrl_t			const *opt_ctrl,
					result_addr_t			*result_addr,
					fc_addr_t			const *fc_addr,
					int32_t 			*user_err,
					lkof_process_type		mode,
					uint64_t			frame_id);

int32_t lkof_hal_trackpoint_one(image_address_t			image_address,
					uint32_t		roi_num_in,
					one_roi_param_t		const *one_roi_param,
					result_addr_t		*result_addr,
					fc_addr_t		const *fc_addr,
					int32_t 		*user_err,
					lkof_process_type	mode,
					uint64_t		frame_id);


int32_t lkof_wait_done(uint64_t mode, uint64_t *frame_id, int32_t *user_err, int32_t timeout);

int32_t lkof_compute_rms(uint32_t roi_num_in, uint32_t const *n_points, uint32_t *size, result_mem_type_t result_mem_type, uint32_t last_roi);

int32_t lkof_compute_fms(uint32_t roi_num_in, opt_ctrl_t const *opt_ctrl, uint32_t *size);

int32_t lkof_get_fifo_status(fifo_info_status_t *fifo_status);

lkof_ctx_t *get_lkof_ctx(void);

#endif //LKOF_CORE_H

