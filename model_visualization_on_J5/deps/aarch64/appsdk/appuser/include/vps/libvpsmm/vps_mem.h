/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2020 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/
#ifndef _VPS_MEM_H_
#define _VPS_MEM_H_
#include <list.h>
#include <hb_vps_mem.h>
#include <local_mem.h>

struct mem_node {
	struct list_head list_head;
	void *id_start;
	vps_mem_hd_t *handle;
	int32_t size;
};

#endif//_VPS_MEM_H_
