/*
 * Horizon Robotics
 *
 * Copyright (C) 2021 Horizon Robotics Inc.
 * All rights reserved.
 * Author:
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef __EMM_DAEMON_API_H__
#define __EMM_DAEMON_API_H__

#include <stdint.h>
#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif
/* 场景模式,如:"ADAS",与json脚本中"scenario"对应 */
#define HB_SCENE_MAX_LEN      32
/* app安装路径最大长度 */
#define HB_PATH_MAX_LEN       128
/* emmd默认服务端口, json配置中可配 */
#define HB_EMM_SERVER_PORT    20900
/* emmd默认本地域名, json配置中可配 */
#define EMM_SERVER_DOMAIN "/tmp/emmd_server_domain"
/* emmd configurable json */
#if !defined(__X86_DEBUG__) && !defined(__ARM64_DEBUG__)
#define HB_EMM_JSON_PATH "/system/etc/emmd/EmmdCfg.json"
#endif
#if !defined(__X86_DEBUG__) && defined(__ARM64_DEBUG__)
#define HB_EMM_JSON_PATH "/system/etc/emmd/EmmdCfgArmTest.json"
#endif
#if defined(__X86_DEBUG__) && !defined(__ARM64_DEBUG__)
#define HB_EMM_JSON_PATH "./cfg/EmmdCfgX86Test.json"
#endif

#ifndef INET_ADDRSTRLEN
#define INET_ADDRSTRLEN 16
#endif

/* emmd支持的服务请求类型 */
typedef enum {
    EMM_REQ_APP_START = 1, /* 向emmd请求启动app_id对应应用程序 */
    EMM_REQ_APP_STOP,      /* 向emmd请求停止app_id对应应用程序 */
    EMM_REQ_MEM_SIZE,      /* 向emmd请求app_id对应应用程序的物理内存使用大小 */
    EMM_REQ_CPU_RATE,      /* 向emmd请求app_id对应应用程序的cpu占用率 */
    EMM_REQ_CPU_THERMAL,   /* 向emmd请求app_id对应应用程序的cpu温度 */
    EMM_REQ_BPU_THERMAL,   /* 向emmd请求app_id对应应用程序的bpu温度 */
    EMM_REQ_HEART_ACK,     /* 向emmd请求定时心跳响应信息 */
    EMM_REQ_SCENE_QUERY,   /* 向emmd请求查询当前模式,如:adas/pilot等 */
    EMM_REQ_SCENE_SWITCH,  /* 向emmd请求模式切换,如:adas/pilot等 */
    EMM_REQ_MAX_NUM,       /* 向emmd一次请求的最大个数 */
    EMM_REQ_INVLID_TYPE = 0xFFFF /* emmd校验失败返回的无效类型 */
} hb_emm_req_type;

/* 唯一标识一个APP */
typedef struct app_id_s {
    /* app运行的soc id */
    uint32_t soc_id;
    /* app安装绝对路径 :与json配置中的name字段对应  */
    char    app_name[HB_PATH_MAX_LEN];
    /* 优先判断:如install_name[0]='/0',使用app_name判断 */
    char    install_name[HB_PATH_MAX_LEN];
} __attribute__((packed)) hb_app_id;

/* 描述单个请求 */
typedef struct emm_req_s {
    /* 向emmd请求的类型: hb_emm_req_type */
    uint32_t type;
    /* 向emmd请求EMM_REQ_SCENE_SWITCH,请求切换到scene对应场景下 */
    char    scene[HB_SCENE_MAX_LEN];
    /* EMM_REQ_CPU_RATE请求cpu占有率时需填写:统计时间间隔ms */
    uint32_t cpu_interval;
    /* emmd回应EMM_REQ_CPU_RATE请求的次数 */
    uint32_t rsp_cpu_count;
    /**
     * emmd回应EMM_REQ_HEART_ACK请求, 上报周期是多少ms
     * 0默认上报一次, 非0定周期上报
     */
    uint32_t heart_ack_cycle;
    /* emmd回应EMM_REQ_HEART_ACK请求的次数 */
    uint32_t rsp_heart_count;
} __attribute__((packed)) hb_emm_req;

typedef struct {
    struct {
        /* 请求业务向emmd请求id,与emmd的回应消息中rsp_no对应 */
        uint32_t                 req_no;
        /* 当向emmd请求app的mem/cpu/bpu/信息时使用 */
        hb_app_id                app_id;
        /**
         * 当emmd与请求业务在不同soc时,需要填写emmd所在soc的ip
         * 远端通信填写目标sever的IP地址, 通信时会先匹配与本地
         * 是否一致, 相同采用本地UNIX通信, 反之采用远端通信
         */
        char                     ipaddr[INET_ADDRSTRLEN];
        /* emmd服务器默认的端口号, 用于客户端进行请求, 大于等于10000 */
        uint16_t                 port;
        /**
         * 请求信息和回应方式的描述, 请求信息和枚举类型对应
         * reqinfo[0...EMM_REQ_MAX_NUM], 支持最大请求个数
         * 约束: 支持一次请求多个不同类型;不支持一次请求多个相同类型
         */
        hb_emm_req               reqinfo[EMM_REQ_MAX_NUM];
        /* 预留长度 */
        uint8_t                  res[16];
    } __attribute__((packed)) req;
    struct {
        /* emmd回应的type: hb_emm_req_type */
        uint32_t                 type;
        /* emmd回应id,与请求业务传入req_no对应 */
        uint32_t                 rsp_no;
        /* emmd回应app的mem/cpu/bpu/请求信息时使用 */
        hb_app_id                app_id;
        /* emmd回应SCENE_QUERY请求和SCENE_SWITCH时填写 */
        char                     scene[HB_SCENE_MAX_LEN];
        /* emmd回应心跳, 仅在心跳报文中使用 */
        uint32_t                 heart_ack;
        /* emmd回应 app_id对应的pid,暂未启用 */
        int32_t                  pid;
        /* emmd回应app_id对应程序的物理内存使用情况KB */
        uint32_t                 phy_mem;
        /* emmd回应app_id对应程序的cpu占用率:%cpurate * 100, 默认放大100倍 */
        uint32_t                 cpu_rate;
        /* emmd回应EMM_REQ_CPU_THERMAL */
        uint32_t                 cputhermal;
        /* emmd回应EMM_REQ_BPU_THERMAL */
        uint32_t                 bputhermal;
        /* emmd对一次req的第几次回复 */
        uint32_t                 rsp_index;
        /* 预留长度 */
        uint8_t                  res[16];
    } __attribute__((packed)) rsp;
} hb_emm_info;

/* emm API errno */
enum emmd_errno {
    HB_RTE_SUCCESS               = 0,
    HB_RTE_ERROR                 = -1,
    HB_ERR_COM_TIMEOUT           = ETIMEDOUT,
    HB_ERR_COM_INVALID           = 0xEE0,
    HB_ERR_MSG_SHORT             = 0xEE00,
    HB_ERR_MSG_MAGIC             = 0xEE01,
    HB_ERR_MSG_INVALID           = 0xEE02,
    HB_ERR_MSG_CHECKSUM          = 0xEE03,
    HB_ERR_MSG_START_APP         = 0xEE04,
    HB_ERR_MSG_STOP_APP          = 0xEE05,
    HB_ERR_MSG_MEM_SIZE          = 0xEE06,
    HB_ERR_MSG_CPU_RATE          = 0xEE07,
    HB_ERR_MSG_CPU_THEMAL        = 0xEE08,
    HB_ERR_MSG_BPU_THEMAL        = 0xEE09,
    HB_ERR_MSG_SCENE_QUERY       = 0xEE0A,
    HB_ERR_MSG_SCENE_SWITCH      = 0xEE0B,
    HB_ERR_MSG_HEART_ACK         = 0xEE0C,
    HB_ERR_MSG_INVALID_MAX       = 0xEEEE
};

/*! cancel a emmd client
 *
 * @param p_handle emmd handle pointer
 * @param p_info request emmd info struct, fill p_info before create
 * @return 0 if successfully recveive, otherwise return -1 of error
 */
int32_t hb_emm_create(void **p_handle, hb_emm_info *p_info);

/*! execute and monitor management emmd
 *
 * @param p_handle emmd handle which created by @hb_emm_create()
 * @param p_info request emmd info struct, fill p_info before request
 * @param err_no if request error, errno HB_ERR_COM_TIMEOUT or other will be set
 * @return 0 if successfully request, otherwise return -1 of error
 */
int32_t hb_emm_request(void *p_handle, hb_emm_info *p_info, int32_t *err_no);

/*! receive reponse info
 *
 * @param p_handle emmd handle which created by @hb_emm_create()
 * @param p_info recv emmd info struct, must be valid
 * @param err_no if receive error, errno @emmd_errno will be set
 * @param timeout receive timeout, blockable if 0 otherwise receive timeout
 * @return 0 if successfully recveive, otherwise return -1 of error
 */
int32_t hb_emm_receive(void *p_handle, hb_emm_info *p_info, int32_t *err_no, uint32_t timeout);

/*! cancel a request
 *
 * @param p_handle emmd handle which created by @hb_emm_create()
 * @return 0 if successfully recveive, otherwise return -1 of error
 */
int32_t hb_emm_cancel(void *p_handle);

#ifdef __cplusplus
}
#endif

#endif
