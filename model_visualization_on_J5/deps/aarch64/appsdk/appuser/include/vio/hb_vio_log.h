/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2020 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/

#ifndef HB_VIO_LOG_H__
#define HB_VIO_LOG_H__

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#include <sys/time.h>

#define VCS_DEBUG (0)

#define vcs_err(format, ...)				\
	do {									\
		char str[30];						\
		struct timespec tp;							\
		(void)clock_gettime(CLOCK_MONOTONIC, &tp);				\
		snprintf(str, sizeof(str), "%ld.%06ld", tp.tv_sec, tp.tv_nsec/1000);				\
		ALOGE("[%s]%s[%d] E: "format"", str, __func__, __LINE__, ##__VA_ARGS__);	\
	} while (0);

#define vcs_warn(format, ...)				\
	do {									\
		char str[30];						\
		struct timespec tp;							\
		(void)clock_gettime(CLOCK_MONOTONIC, &tp);				\
		snprintf(str, sizeof(str), "%ld.%06ld", tp.tv_sec, tp.tv_nsec/1000);				\
		ALOGW("[%s]%s[%d] W: "format"", str, __func__, __LINE__, ##__VA_ARGS__);	\
	} while (0);

#define vcs_log(format, ...)				\
	do {									\
		char str[30];						\
		struct timespec tp;							\
		(void)clock_gettime(CLOCK_MONOTONIC, &tp);				\
		snprintf(str, sizeof(str), "%ld.%06ld", tp.tv_sec, tp.tv_nsec/1000);				\
		ALOGI("[%s]%s[%d] L: "format"", str, __func__, __LINE__, ##__VA_ARGS__); \
	} while (0);

#if VCS_DEBUG
#define vcs_dbg(format, ...)				\
	do {									\
		char str[30];						\
		struct timespec tp;							\
		(void)clock_gettime(CLOCK_MONOTONIC, &tp);				\
		snprintf(str, sizeof(str), "%ld.%06ld", tp.tv_sec, tp.tv_nsec/1000);				\
		ALOGI("[%s]%s[%d] D: "format"", str, __func__, __LINE__, ##__VA_ARGS__);	\
	} while (0);
#else
#define vcs_dbg(format, ...)
#endif


#ifdef __cplusplus
}
#endif

#endif // HB_VIO_LOG_H__
