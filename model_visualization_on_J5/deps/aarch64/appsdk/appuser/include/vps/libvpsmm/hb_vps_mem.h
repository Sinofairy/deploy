/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2020 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/
#ifndef _HB_VPS_MEM_H_
#define _HB_VPS_MEM_H_
#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define vps_addr_t uint64_t
//bit 0 CACHE:1 or NO CACHE:0
#define VPS_NON_CACHEABLE (0 << 0)
#define VPS_CACHEABLE (1 << 0)
//bit 1 DDR:0 or SRAM:1
#define VPS_FROM_DDR (0 << 1)
#define VPS_FROM_SRAM (1 << 1)
//FLUSH CACHE
#define VPS_MEM_INVALIDATE (1)
#define VPS_MEM_CLEAN (2)


//ERROR CODE
#define VPS_MEM_ERR_BASE  (10000u)
#define HB_VPS_MEM_OK			(0)
#define	HB_VPS_MEM_BUFFER_NO		(VPS_MEM_ERR_BASE + 1)
#define HB_VPS_MEM_PARTERM_INV		(VPS_MEM_ERR_BASE + 2)
#define	HB_VPS_MEM_CFG_ATTR_INV		(VPS_MEM_ERR_BASE + 3)
#define	HB_VPS_MEM_VADDR_INV		(VPS_MEM_ERR_BASE + 4)
#define	HB_VPS_MEM_PADDR_INV		(VPS_MEM_ERR_BASE + 5)
#define	HB_VPS_MEM_POINTER_NULL		(VPS_MEM_ERR_BASE + 6)
#define	HB_VPS_MEM_MEM_HD_NULL		(VPS_MEM_ERR_BASE + 7)
#define	HB_VPS_MEM_MEM_INIT_NOT_OK	(VPS_MEM_ERR_BASE + 8)
#define	HB_VPS_MEM_MEM_ALREADY_FREE	(VPS_MEM_ERR_BASE + 9)


void *hb_vps_mem_alloc(uint64_t *paddr, uint32_t size, uint32_t flag);
int32_t hb_vps_mem_free(void *vaddr);
int32_t hb_vps_mem_is_cacheable(void *vaddr);
int32_t hb_vps_mem_cache_flush(void *vaddr, uint32_t size, uint32_t flag);
int32_t hb_vps_memcpy(void *dst_addr, void *src_addr, uint32_t size);
void print_mem_handle_info(void);

#ifdef __cplusplus
}
#endif

#endif //_HB_VPS_MEM_H_
