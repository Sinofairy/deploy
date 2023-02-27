/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2020 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/
#ifndef _LIST_H_
#define _LIST_H_
#include <pthread.h>

struct list_head {
	int *p_counter;
	pthread_mutex_t *lock;
	struct list_head *next, *prev;
};

void init_list_head(struct list_head *list,
		unsigned int *counter, pthread_mutex_t *lock);

void list_add(struct list_head *new_node, struct list_head *head);
void list_add_tail(struct list_head *new_node, struct list_head *head);
void list_add_before(struct list_head *new_node, struct list_head *head);

void list_del(struct list_head *entry);
int list_is_last(const struct list_head *list,
				const struct list_head *head);

int list_empty(const struct list_head *head);
unsigned int list_length(struct list_head *head);

#define list_first(head) ((head)->next)
#define list_last(head) ((head)->prev)

#define list_for_each(pos, head) \
	for (pos = (head)->next; pos != (head); pos = pos->next)

#define list_for_each_safe(pos, n, head) \
	for (pos = (head)->next, n = pos->next; pos != (head); \
		pos = n, n = pos->next)

#endif
