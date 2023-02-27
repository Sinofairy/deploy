#ifndef WAKE_LOCK_H
#define WAKE_LOCK_H

#ifdef __cplusplus
extern "C" {
#endif

struct hb_wakelock {
	unsigned long data;
};

enum hb_wakelock_type {
	WAKE_LOCK_SUSPEND,
	WAKE_LOCK_IDLE,
};

/*
 * This wakelock api doesn't support permission management yet, so use
 * it with special care.
 * only process with CAP_BLOCK_SUSPEND permission can use this api,
 * otherwise it will failed.
 */


/**
 * hb_wakelock_init - Initialize a hobot wakelock
 * @lock: pointer of a hb_wakelock object, must alloc it first.
 * @type: type in @hb_wakelock_type
 * @name: Name for indentical.
 *
 * hb_wakelock_init will initialize a hb_wakelock struct, this
 * will be needed for later lock & unlock operation.
 *
 * Return: = 0 on success, < 0 in case of error.
 */
int hb_wakelock_init(struct hb_wakelock *lock, int type, const char *name);

/**
 * hb_wakelock_lock - Acquire a perflock for specific time
 * @lock: hb_wakelock object.
 * @timeout: timeout in nanoseconds
 *
 * hb_wakelock_lock will get a lock from kernel, to prevent system
 * going to specified state.
 *
 * Return: = 0 on success, < 0 in case of error.
 */
int hb_wakelock_lock(struct hb_wakelock *lock, unsigned long timeout);

/**
 * hb_wakelock_unlock - Release a perflock for specific time
 * @lock: hb_wakelock object.e
 *
 * hb_wakelock_unlock will release a lock from kernel.
 *
 * Return: = 0 on success, < 0 in case of error.
 */
int hb_wakelock_unlock(struct hb_wakelock *lock);

/**
 * hb_wakelock_destory - Release every thing related to this lock.
 * @lock: hb_wakelock object.
 *
 * hb_wakelock_unlock will release everything related to the @lock.
 *
 * Return: = 0 on success, < 0 in case of error.
 */
int hb_wakelock_destory(struct hb_wakelock *lock);

#ifdef __cplusplus
}
#endif

#endif
