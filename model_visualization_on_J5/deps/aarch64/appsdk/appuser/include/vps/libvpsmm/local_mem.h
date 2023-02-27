/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2020 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/
#ifndef _LOCAL_MEM_H_
#define _LOCAL_MEM_H_
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <ion.h>

#define ALIGN_UP(a, size)    ((a+size-1) & (~ (size-1)))
#define ALIGN_DOWN(a, size)  (a & (~(size-1)) )
#define ALIGN_UP4(d)   ALIGN_UP(d, 4)
#define ALIGN_UP16(d)  ALIGN_UP(d, 16)
#define ALIGN_UP64(d)  ALIGN_UP(d, 64)

#define ALIGN_DOWN4(d)    ALIGN_DOWN(d, 4)
#define ALIGN_DOWN16(d)   ALIGN_DOWN(d, 16)
#define ALIGN_DOWN64(d)   ALIGN_DOWN(d, 64)
#define SCALE3_2(a)     ((a*3)>>1)

struct mem_id {
	//struct ion_handle *ion_hd;
	ion_user_handle_t ion_hd;
	int32_t map_fd;
	void *tmp_paddr;
	int32_t cached;
	int32_t sram;
};

typedef struct vps_mem_handle {
	void *vaddr;
	void *paddr;
	uint32_t size;
	int32_t flag;
	void *mem_id;
}vps_mem_hd_t;

int32_t ion_init(void);
vps_mem_hd_t *vps_mem_alloc(size_t len, uint32_t flag);
vps_mem_hd_t *vps_mem_alloc_range(size_t len, uint32_t flag);
int32_t vps_mem_free(vps_mem_hd_t *mem_handle);
int32_t vps_mem_cache_flush(vps_mem_hd_t *mem_handle, uint32_t start_offset, uint32_t size);
int32_t vps_mem_cache_disable(vps_mem_hd_t *mem_handle, uint32_t start_offset, uint32_t size);

void *vps_mem_vaddr(vps_mem_hd_t *mem_handle);
void *vps_mem_paddr(vps_mem_hd_t *mem_handle);

int32_t ion_buffer_map(int32_t size, int32_t fd, char **addr);
int32_t ion_alloc_phy(int32_t size, int32_t *fd, char **vaddr, uint64_t  *paddr);
int32_t ion_buffer_free(int32_t *fd, int32_t size, char **addr, _Bool need_map);

#endif//_LOCAL_MEM_H_
