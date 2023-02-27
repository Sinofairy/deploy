#ifndef VIO_MONTOR_H
#define VIO_MONTOR_H

#include <stdint.h>
#include "server_cmd.h"

#ifdef __cplusplus
extern "C"
{
#endif

int vio_montor_start(const char *pathname, uint32_t gdc_rotation, uint32_t video, uint32_t stats);
int vio_montor_stop(void);
int vio_montor_start_lite(const char *pathname);
void vio_montor_stop_lite(void);
void send_yuvdata(struct cmd_header *info, void *ptr, uint32_t size, void *ptr1, uint32_t size1);

#ifdef __cplusplus
	}
#endif


#endif // COMMON_H
