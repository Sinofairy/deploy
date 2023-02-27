/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2020 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/
#ifndef LKOF_INSTLIST_H
#define LKOF_INSTLIST_H
#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

typedef struct instlist_addr_s {
	int32_t		fd;
	int32_t		size;
	uint64_t	pa_inst_addr;
	char		*virt_inst_addr;
}instlist_addr_t;

typedef struct instlist_ctx_s {
	int32_t			dev_fd;
	char 			inslist_path[100];
	int32_t 			inited;
	instlist_addr_t 	instlist_addr;
}instlist_ctx_t;

instlist_ctx_t *lkof_init_instlist(int32_t fd);
int32_t lkof_free_instlist(instlist_ctx_t *instlist_ctx);

#endif //LKOF_INSTLIST_H
