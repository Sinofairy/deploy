/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2020 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/
#ifndef WAIT_MASK_H
#define WAIT_MASK_H
#include <stdint.h>
#include <pthread.h>
#include <list_safe.h>
#include <lkof_core.h>

typedef enum {
	WAIT_TYPE_ASYNC,
	WAIT_TYPE_FRAMEID,
	WAIT_TYPE_MAX,
}wait_type;

struct wait_mask {
	struct list_safe_head node;
	int8_t type;
	uint64_t frameid;

	struct param_fifo *done_params;

	pthread_mutex_t poll_lock;
	pthread_condattr_t cond_attr;
	pthread_cond_t wait_poll_cond;
};

struct wait_mask *wait_mask_get(wait_type type, uint64_t frameid);
struct wait_mask *wait_mask_bind(wait_type type, uint64_t frameid);
int32_t wait_mask_del_by_frameid(wait_type type, uint64_t frameid);

int32_t wait_mask_clear(void);
void printf_wait_mask_node(lkof_ctx_t const *ctx);

#endif //WAIT_MASK_H
