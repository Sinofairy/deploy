/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2020 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/
#ifndef J5_LKOF_INCLUDE_LIST_SAFE_H_
#define J5_LKOF_INCLUDE_LIST_SAFE_H_
#include <pthread.h>
#include <stdint.h>

struct list_safe_head {
	int32_t *p_counter;
	pthread_mutex_t *lock;
	struct list_safe_head *next, *prev;
};

void init_list_safe_head(struct list_safe_head *list_safe,
		uint32_t *counter, pthread_mutex_t *lock);

void list_safe_add(struct list_safe_head *new_node, struct list_safe_head *head);
void list_safe_add_tail(struct list_safe_head *new_node, struct list_safe_head *head);
void list_safe_add_before(struct list_safe_head *new_node, struct list_safe_head *head);

void list_safe_del(struct list_safe_head *entry);
int32_t list_safe_is_last(const struct list_safe_head *list_safe,
				const struct list_safe_head *head);

int32_t list_safe_empty(const struct list_safe_head *head);
int32_t list_safe_length(struct list_safe_head *head);

#define list_safe_first(head) ((head)->next)
#define list_safe_last(head) ((head)->prev)

#define list_safe_for_each(pos, head) \
	for (pos = (head)->next; pos != (head); pos = pos->next)

#define list_safe_for_each_safe(pos, n, head) \
	for (pos = (head)->next, n = pos->next; pos != (head); \
		pos = n, n = pos->next)

#endif //J5_LKOF_INCLUDE_LIST_SAFE_H_
