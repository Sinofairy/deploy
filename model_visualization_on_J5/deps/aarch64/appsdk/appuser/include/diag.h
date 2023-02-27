/***************************************************************************
 *   Copyright (C) 2019 by horizon.                                        *
 *   bo01.chen@horizon.ai                                                  *
 *                                                                         *
 *   Diag header file.                                                     *
 *                                                                         *
 ***************************************************************************/
#ifndef DIAG_H
#define DIAG_H

#ifdef __CPLUSPLUS
extern "C" {
#endif

#include <errno.h>
#include <fcntl.h>
#include <linux/netlink.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef __CPLUSPLUS
}
#endif

#ifdef DIAG_GLOBALS
#define DIAG_EXT
#else
#define DIAG_EXT extern
#endif

typedef enum {
	DIAG_REL_NULL,
	DIAG_REL_DEBUG,
	DIAG_REL_SPI_DEBUG,
	DIAG_REL_ZMQ_DEBUG,
	DIAG_REL_RELEASE,
	DIAG_REL_END
} DIAG_REL_MODE;

DIAG_EXT DIAG_REL_MODE DiagRelMode;

typedef enum {
	SPI_SW_NULL,
	SPI_SW_OFF,
	SPI_SW_ON,
	SPI_SW_END
} SPI_SW_TBL;
DIAG_EXT SPI_SW_TBL spi_switch_sts;

typedef struct {
	uint8_t bitmask_period;
	uint8_t enable_file;
} GLOBAL_CONFIG;
DIAG_EXT GLOBAL_CONFIG global_config;

#define ENABLE_FILE_RECORD 1
#define DISABLE_FILE_RECORD 0
//#define DEBUG_TEST
#define RELEASE_SOFT

/* err code status */
#define ERR_RECOVERY 2
#define ERR_OCCUR 3
#define ERR_CANCEL_TRIGGED 0
#define ERR_RECOVERY_TRIGGED 1
#define ERR_OCCUR_TRIGGED 2

/* to be confirm */
#define TOTAL_APP_MODULE_NUM 5

typedef enum {
	J2SOC_OK = 0,
	J2SOC_ALREADY_INITED = 1,
	J2SOC_FAIL = -1,
	J2SOC_INVALID_REQ = -2,
	J2SOC_NOT_INITED = -3,
	J2SOC_INVALID_ENGINE_TYPE = -4,
	J2SOC_INPUT_BUFFER_FULL = -5,
	J2SOC_INVALID_PARAM = -6,
	J2SOC_TIMEOUT = -7,

	J2SOC_AUTH_CONN_FAIL = -100,
	J2SOC_AUTH_OFFLINE_FAIL = -101,
	J2SOC_AUTH_ONLINE_FAIL = -102,

	J2SOC_NETWORK_ERR = -106,

	J2SOC_SYSTIME_ERR = -107,
} J2SOC_STATUS;

#define UDS_TIME_STEP 50  // 50ms

typedef struct DTC_Snapshot_Data_Type_ {
	uint16_t Env_len;  // env data len
	uint8_t *pEnv;     // env data from dtc file
} DTC_Snapshot_Data_Type;
DIAG_EXT DTC_Snapshot_Data_Type DTC_Snapshot_Data_Map_Type;

#define FIRST_ERROR_RECORD_NUM 0x00
#define LAST_ERROR_RECORD_NUM 0x01
#define ERR_TO_SUCCESS_RECORD_NUM 0x02

extern pthread_mutex_t env_table_mutex;
extern pthread_mutex_t dtc_info_mutex;
extern pthread_mutex_t dtc_sts_mutex;

// extern TDisg_status diag_msg_app_unmask_id_set(struct diag_unmask_id
// *unmask_id);

DIAG_EXT int specify_dtc_handle_process(uint16_t pdtc_id, uint8_t pdtc_status,
																				char const *pdtc_desp);
DIAG_EXT int general_env_handle_process(uint16_t pdtc_id, uint8_t pdtc_status,
																				char const *pdtc_desp);
DIAG_EXT void dtc_list_management(void);
DIAG_EXT int diagnose_init(void);
DIAG_EXT int diagnose_deinit(void);
DIAG_EXT int spi_thread_init(void);

#endif
