#ifndef HOBOT_PERF_H
#define HOBOT_PERF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NR_CPUS 32
#define NR_BPUS 32
#define NR_DMCS 8

typedef enum {
	PERFLOCK_ID_CPU = 0,
	PERFLOCK_ID_CPU_MAX = PERFLOCK_ID_CPU + NR_CPUS,
	PERFLOCK_ID_CPUHP,
	PERFLOCK_ID_BPU,
	PERFLOCK_ID_BPU_MAX = PERFLOCK_ID_BPU + NR_BPUS,
	PERFLOCK_ID_DMC,
	PERFLOCK_ID_DMC_MAX = PERFLOCK_ID_DMC + NR_DMCS,
	PERFLOCK_ID_MAX,
} perlock_event;

struct perf_lock_event {
	perlock_event item;
	int min;
	int max;
};

#define PERFLOCK_TIMEOUT_NONE     (0)

typedef struct hb_pl_req_handle *hb_pl_handle;

/**
 * hb_perflock_acq - Acquire a perflock for specific time
 * @list: List of perflock event, it value from enum @perflock_event.
 * @len: Length of perflock event list.
 * @timeout: Timeout, in microseconds.
 *
 * hb_perflock_acq will take a lock to ensure the CPU, BPU etc core
 * running at a high level, it takes an list of predefined interger
 * value to indicate how much CPU resource need to acquire
 *
 * Return: NULL on failed, success otherwise.
 */
hb_pl_handle hb_perflock_acq(struct perf_lock_event events[], int len, int timeout);

/**
 * hobot_perflock_rel - Release a perflock
 * @handle: return value from @hb_perflock_acq
 * @len: Length of perflock event list.
 * @timeout: Timeout, in microseconds.
 *
 * hobot_perflock_rel will release a perflock, if this perflock have
 * expired is activate time, it will do nothing, just to free
 * everything alloced when @hb_perflock_acq. After this function
 * successfully returned, you shouldn't use @handle anymore.
 *
 * Return: 0
 */
int hb_perflock_rel(hb_pl_handle handle);

#ifdef __cplusplus
}
#endif

#endif
