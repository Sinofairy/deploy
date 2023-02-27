/*
 *    COPYRIGHT NOTICE
 *    Copyright 2020 Horizon Robotics, Inc.
 *    All rights reserved.
 */
#ifndef SPI_SERVICE_LOG_H
#define SPI_SERVICE_LOG_H
#include <linux/types.h>
#include <log.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LOG_TAG "[SPI_SERVICE]"
#define STRINGIZE_NO_EXPANSION(x) #x
#define STRINGIZE(x) STRINGIZE_NO_EXPANSION(x)
#define HERE __FILE__ ":" STRINGIZE(__LINE__)
#define L_INFO "[SPI_SERVICE_INFO][" HERE "] "
#define L_WARNING "[SPI_SERVICE_WARNING]" HERE "] "
#define L_ERROR "[SPI_SERVICE_ERROR][" HERE "] "
#define L_DEBUG "[SPI_SERVICE_DEBUG][" HERE "] "
#define L_FILE "[SPI_SERVICE_FILE][" HERE "] "
#define FILE_NAME_HEAD "/userdata/spi_service_log"
void SPI_SERVICE_PRINT(const char *fmt, ...);
void SPI_SERVICE_LOGERR(const char *fmt, ...);
void SPI_SERVICE_LOGINFO(const char *fmt, ...);
void SPI_SERVICE_LOGFILE(const char *fmt, ...);
void SPI_SERVICE_LOGDEBUG(int32_t level, const char *fmt, ...);
void SPI_SERVICE_print(char *level, const char *fmt, ...);
static inline int32_t check_debug_level(void) {
  static int32_t debug_flag = -1;
  const char *dbg_env;
  int32_t ret;

  if (debug_flag >= 0) {
    return debug_flag;
  } else {
    dbg_env = getenv("SPI_SERVICE_DEBUG_LEVEL");
    if (dbg_env != NULL) {
      ret = atoi(dbg_env);
      if (ret <= 0) {
        debug_flag = 0;
      } else {
        debug_flag = ret;
      }
    } else {
      debug_flag = 0;
    }
  }
  return debug_flag;
}

#ifndef pr_fmt
#define pr_fmt(fmt) fmt
#endif

#define SPI_SERVICE_LOGERR(fmt, ...)                        \
  do {                                                      \
    fprintf(stderr, L_ERROR "" pr_fmt(fmt), ##__VA_ARGS__); \
    ALOGE(L_ERROR "" fmt, ##__VA_ARGS__);                   \
  } while (0);
#define SPI_SERVICE_LOGINFO(fmt, ...)                      \
  do {                                                     \
    fprintf(stdout, L_INFO "" pr_fmt(fmt), ##__VA_ARGS__); \
    ALOGI(L_INFO "" fmt, ##__VA_ARGS__);                   \
  } while (0);
#define SPI_SERVICE_LOGDEBUG(fmt, ...)                        \
  do {                                                        \
    int loglevel = check_debug_level();                       \
    if (loglevel == 3) {                                      \
      fprintf(stdout, L_DEBUG "" pr_fmt(fmt), ##__VA_ARGS__); \
      ALOGD(L_DEBUG "" fmt, ##__VA_ARGS__);                   \
    }                                                         \
  } while (0);
#define SPI_SERVICE_LOGFILE(fmt, ...)                      \
  do {                                                     \
    FILE *fp = fopen(FILE_NAME_HEAD, "a+");                \
    if (fp == NULL) break;                                 \
    fprintf(fp, L_FILE "" pr_fmt(fmt), ##__VA_ARGS__);     \
    fprintf(stdout, L_FILE "" pr_fmt(fmt), ##__VA_ARGS__); \
    ALOGI(L_FILE "" fmt, ##__VA_ARGS__);                   \
    fclose(fp);                                            \
  } while (0);
#endif
