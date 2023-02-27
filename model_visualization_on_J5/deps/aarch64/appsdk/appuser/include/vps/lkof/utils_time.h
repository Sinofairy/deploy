/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2020 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/
#ifndef UTILS_TIME_H
#define UTILS_TIME_H
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>/*PRQA S ALL*/

#define ONE_SEC_TO_MS (1000)
#define ONE_SEC_TO_NS (1000000)
int32_t load_file_to_buff(const char *path, char *filebuff, int32_t size);
void time_add_ms(struct timeval *time, uint16_t ms);
int32_t get_cost_time_ms(struct timeval const *start, struct timeval const *end);
int32_t cond_timedwait_msecs(pthread_cond_t *cond,
		pthread_mutex_t * lock, long msecs);/*PRQA S ALL*/
uint64_t get_time_ms(void);
#endif //UTILS_TIME_H
