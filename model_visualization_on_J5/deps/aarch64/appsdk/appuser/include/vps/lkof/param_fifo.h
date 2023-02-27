/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2020 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/
#ifndef PARAM_FIFO_H
#define PARAM_FIFO_H
#include <stdint.h>
#include <pthread.h>
#include <list_safe.h>
#include <lkof_device.h>
#include <hb_lkof_interface.h>

//#define MAX_FIFO_SIZE (1024)
#define MAX_FIFO_SIZE (8912)

typedef struct list_safe_head fifo_t;

struct param_fifo {
	fifo_t fifo_head;
	uint32_t fifo_counter;
	int32_t cap;
	pthread_mutex_t mutex_lock;
	char name[100];
};

struct param_fifo_node {
	fifo_t node;
	struct lkof_settings_hw_s user_settings_hw;

	struct wait_mask *wait_mask;

	struct timeval step_tv[4];
};

struct param_fifo *param_fifo_create(char *name);
void param_fifo_destory(struct param_fifo *fifo);

struct param_fifo_node *param_fifo_node_create(void);
void param_fifo_node_destory(struct param_fifo_node *fifo_node);

int32_t param_fifo_all_cap(struct param_fifo const *fifo);
int32_t param_fifo_avl_cap(struct param_fifo *fifo);
int32_t param_fifo_empty(struct param_fifo const *fifo);

int32_t param_fifo_push(struct param_fifo *fifo, struct param_fifo_node *fifo_node);
struct param_fifo_node *param_fifo_pop(struct param_fifo *fifo);
struct param_fifo_node *param_fifo_pop_node(struct param_fifo *fifo, uint64_t frame_id);
int32_t param_fifo_check_frameid_is_unique(struct param_fifo *fifo, uint64_t frame_id);

int32_t param_fifo_rollback(struct param_fifo *fifo, int32_t num);
void printf_param_fifo_node(struct param_fifo *fifo);

#endif//PARAM_FIFO_H
