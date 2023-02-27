/*
 *  ion.c
 *
 * Memory Allocator functions for ion
 *
 *   Copyright 2011 Google, Inc
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef SYS_CORE_ION_H
#define SYS_CORE_ION_H

#include <stdint.h>
#include "linux/ion.h"

__BEGIN_DECLS

int32_t ion_open(void);
int32_t ion_close(int32_t fd);
int32_t ion_alloc(int32_t fd, size_t len, size_t align, uint32_t heap_mask,
			uint32_t flags, ion_user_handle_t *handle);
int32_t ion_alloc_fd(int32_t fd, size_t len, size_t align, uint32_t heap_mask,
			uint32_t flags, int32_t *handle_fd);
int32_t ion_sync_fd(int32_t fd, int32_t handle_fd);
int32_t ion_free(int32_t fd, ion_user_handle_t handle);
int32_t ion_map(int32_t fd, ion_user_handle_t handle, size_t length, int32_t prot,
			int32_t flags, off_t offset, unsigned char **ptr, int32_t *map_fd);
int32_t ion_share(int32_t fd, ion_user_handle_t handle, int32_t *share_fd);
int32_t ion_import(int32_t fd, int32_t share_fd, ion_user_handle_t *handle);
int32_t ion_phys(int32_t fd, ion_user_handle_t handle, void **paddr, size_t *len);
int32_t ion_cache_invalid(int32_t fd, ion_user_handle_t handle, void *paddr, void *vaddr, size_t len);
int32_t ion_cache_flush(int32_t fd, ion_user_handle_t handle, void *paddr, void *vaddr, size_t len);
int32_t ion_memcpy(int32_t fd, ion_user_handle_t handle, void *dst_paddr, void *src_paddr, size_t len);

__END_DECLS

#endif /* __SYS_CORE_ION_H */
