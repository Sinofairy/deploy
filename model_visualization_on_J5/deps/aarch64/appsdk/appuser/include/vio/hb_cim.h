#ifndef J5_HB_VIN_CIM_H_
#define J5_HB_VIN_CIM_H_

#include <stdint.h>
#include "cJSON.h"
#include "hb_utils.h"

#define HB_VIN_PIPELINE_MAX 36u

#define MIPI_RX0	0u
#define MIPI_RX1	1u
#define MIPI_RX2	2u
#define MIPI_RX3	3u

typedef struct cim_json_s {
	cJSON *root;
	cJSON *cim;
	cJSON *cim_dma;
} cim_json_t;

#ifndef NOCIM
int32_t hb_vin_cim_init(uint32_t pipeline_id, const char *data_path, uint32_t mipi_rx);
int32_t hb_vin_cim_start(uint32_t pipeline_id, uint32_t mipi_rx);
int32_t hb_vin_cim_stop(uint32_t pipeline_id, uint32_t mipi_rx);
int32_t hb_vin_cim_deinit(uint32_t pipeline_id, uint32_t mipi_rx);
int32_t hb_vin_cim_bypass_enable(uint32_t pipeline_id, uint32_t mipi_rx, uint32_t enable);
int32_t hb_vin_cim_set_fps_ctrl(uint32_t pipeline_id,  uint32_t mipi_rx,
		uint32_t skip_frame, uint32_t in_fps, uint32_t out_fps);
int32_t hb_vin_cim_get_data(uint32_t pipeline_id, CAM_DATA_TYPE_E data_type, void* data);
int32_t hb_vin_cim_put_data(uint32_t pipeline_id, CAM_DATA_TYPE_E data_type, const void* data);
#else
inline int32_t hb_vin_cim_init(uint32_t pipeline_id, const char *data_path, uint32_t mipi_rx) { return 0; }
inline int32_t hb_vin_cim_start(uint32_t pipeline_id, uint32_t mipi_rx) { return 0; }
inline int32_t hb_vin_cim_stop(uint32_t pipeline_id, uint32_t mipi_rx) { return 0; }
inline int32_t hb_vin_cim_deinit(uint32_t pipeline_id, uint32_t mipi_rx) { return 0; }
inline int32_t hb_vin_cim_bypass_enable(uint32_t pipeline_id, uint32_t mipi_rx, uint32_t enable) { return 0; }
inline int32_t hb_vin_cim_set_fps_ctrl(uint32_t pipeline_id,  uint32_t mipi_rx,
		uint32_t skip_frame, uint32_t in_fps, uint32_t out_fps) { return 0; }
inline int32_t hb_vin_cim_get_data(uint32_t pipeline_id, CAM_DATA_TYPE_E data_type, void* data) { return 0;}
inline int32_t hb_vin_cim_put_data(uint32_t pipeline_id, CAM_DATA_TYPE_E data_type, const void* data) { return 0;}
#endif

#endif
