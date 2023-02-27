/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2020 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/
#ifndef _VPS_LOG_H_
#define _VPS_LOG_H_
#include <stdio.h>
#include <stdint.h>

#define VPS_LOG_PRINT 0
#define print_err(format, ...) printf("[%s]%s[%d] E: "format"\n",__TIME__, __func__, __LINE__, ##__VA_ARGS__)
#define print_log(format, ...) printf("[%s]%s[%d] W: "format"\n",__TIME__, __func__, __LINE__, ##__VA_ARGS__)
#define print_dbg(format, ...) printf("[%s]%s[%d] D: "format"\n",__TIME__, __func__, __LINE__, ##__VA_ARGS__)

#define vps_err(format, ...) printf("vps [%s]%s[%d] E: "format"\n",__TIME__, __func__, __LINE__, ##__VA_ARGS__)
#define vps_warn(format, ...) printf("vps [%s]%s[%d] W: "format"\n",__TIME__, __func__, __LINE__, ##__VA_ARGS__)
#define vps_dbg(format, ...) printf("vps [%s]%s[%d] D: "format"\n",__TIME__, __func__, __LINE__, ##__VA_ARGS__)

#define VPS_ERR(format, ...) printf("vps [%s]%s[%d] E: "format"\n",__TIME__, __func__, __LINE__, ##__VA_ARGS__)
#define VPS_WARN(format, ...) printf("vps [%s]%s[%d] W: "format"\n",__TIME__, __func__, __LINE__, ##__VA_ARGS__)
#define VPS_DEBUG(format, ...) printf("vps [%s]%s[%d] D: "format"\n",__TIME__, __func__, __LINE__, ##__VA_ARGS__)
#endif //_VPS_LOG_H_

