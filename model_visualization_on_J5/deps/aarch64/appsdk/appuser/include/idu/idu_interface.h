/* Copyright: Horizon Robotics
 * Owner: rui.guo@horizon.ai
 * Year : 2019/8/10
 */
#ifndef _IAR_INTERFACE_H
#define _IAR_INTERFACE_H	/*PRQA S ALL*/

#ifdef __cplusplus
extern "C" {
#endif                         /* __cplusplus */

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/fb.h>
#include <stdint.h>

#ifdef XJ3_WB
#include "hb_vio_interface.h"
#include "hb_vio_buffer_mgr.h"
#include "hb_utils.h"
#include "dev_ioctl.h"
#endif

/***********************************************************
 ** 		   Macro global var and struction			  **
***********************************************************/
typedef float	float32_t;
typedef double	float64_t;

#define IDU_DEV_PATH	"/dev/idu_cdev"
#define IDU0_DEV_PATH	"/dev/idu0_cdev"
#define IDU1_DEV_PATH	"/dev/idu1_cdev"

#define IDU_CFG_PATH "/etc/idu/idu.json"

#define ENABLE 1
#define DISABLE 0

#define BIT(nr)	(1UL << (nr))

#define CLEAR(x)	memset(&(x), 0, sizeof(x))

#define HB_IDU_INVALID_CHN_FMT		(10004)
#define HB_IDU_OPEN_DEVICE_FAIL		(10005)
#define HB_IDU_CLOSE_DEVICE_FAIL 	(10006)
#define HB_IDU_NOT_INIT			(10000)
#define HB_IDU_INPUT_POINTER_NULL	(10001)
#define HB_IDU_GET_OUT_CFG_FAIL		(10007)
#define HB_IDU_SET_OUT_CFG_FAIL		(10008)
#define HB_IDU_INVALID_PAR_RANGE	(10002)
#define HB_IDU_INVALID_CHN_NO		(10009)
#define HB_IDU_SET_GAMMA_CFG_FAIL	(10010)
#define HB_IDU_SET_UPSCALE_CFG_FAIL	(10011)
#define HB_IDU_SET_CHN_CFG_FAIL		(10012)
#define HB_IDU_INVALID_IDU_ID		(10003)
#define HB_IDU_ALREADY_INIT		(10013)
#define HB_IDU_INIT_DEV_FAIL		(10014)
#define HB_IDU_PARSER_CFG_FILE_FAIL	(10015)
#define HB_IDU_GET_FRAME_SIZE_FAIL	(10016)
#define HB_IDU_SET_TIMING_FAIL		(10017)
#define HB_IDU_START_FAIL		(10018)
#define HB_IDU_STOP_FAIL		(10019)
#define HB_IDU_GET_STOP_CNT_FAIL	(10020)
#define HB_IDU_GET_START_CNT_FAIL       (10021)
#define HB_IDU_LAYER_ON_FAIL		(10022)
#define HB_IDU_LAYER_OFF_FAIL		(10023)
#define HB_IDU_GET_DISP_DONE_FAIL	(10024)
#define HB_IDU_SET_DISP_ADDR_FAIL	(10025)
#define HB_IDU_OPEN_CFG_FILE_FAIL	(10026)
#define HB_IDU_GET_DEV_INFO_FAIL	(10027)
#define HB_IDU_CHECK_DEV_MODE_FAIL	(10028)
#define HB_IDU_SET_PIX_CLK_FAIL		(10029)
#define HB_IDU_SET_IPI_CLK_FAIL		(10030)
#define HB_IDU_INIT_CHN1_CFG_FAIL	(10031)
#define HB_IDU_INIT_CHN2_CFG_FAIL       (10032)
#define HB_IDU_INIT_CHN3_CFG_FAIL       (10033)
#define HB_IDU_INIT_CHN4_CFG_FAIL       (10034)
#define HB_IDU_CHN_SIZE_LARGE_THAN_IMAGE_SIZE	(10035)
#define HB_IDU_CHN_SIZE_SMALL_THAN_IMAGE_SIZE	(10036)
enum {
	IDU_CHANNEL_1 = 0,
	IDU_CHANNEL_2 = 1,
	IDU_CHANNEL_3 = 2,
	IDU_CHANNEL_4 = 3,
	IDU_CHANNEL_MAX = 4,
};
enum {
	IDU_PRI_1 = 0,
	IDU_PRI_2 = 1,
	IDU_PRI_3 = 2,
	IDU_PRI_4 = 3,
	IDU_PRI_MAX = 4,
};
enum config_type {
	USER_CFG,
	CHANNEL_CFG,
	OVERLAY_CFG,
	SCALE_CFG,
	GAMMA_CFG,
	OUTPUT_CFG,
};

struct type_idu_mode_t {
	uint32_t memory_mode;
	uint32_t output_mode;
	uint32_t channel_enable;
	uint32_t panel_width;
	uint32_t panel_height;
};

#define MAX_FRAME_BUF_SIZE	(1920u*1080u*4u)

typedef unsigned int	phys_addr_t;

struct type_frame_info_t {
	void *addr;
	uint32_t size;
};

enum format_yuv_e {
	FORMAT_YUV422_UYVY	  = 0,
	FORMAT_YUV422_VYUY	  = 1,
	FORMAT_YUV422_YUYV	  = 2,
	FORMAT_YUV422_YVYU	  = 3,
	FORMAT_YUV422SP_UV	  = 4,
	FORMAT_YUV422SP_VU	  = 5,
	FORMAT_YUV420SP_UV	  = 6,
	FORMAT_YUV420SP_VU	  = 7,
	FORMAT_YUV422P_UV	  = 8,
	FORMAT_YUV422P_VU	  = 9,
	FORMAT_YUV420P_UV	  = 10,
	FORMAT_YUV420P_VU	  = 11,
};

enum format_rgb_e {
	FORMAT_8BPP 		  = 0,
	FORMAT_RGB565		  = 1,
	FORMAT_RGB888		  = 2,
	FORMAT_RGB888P		  = 3,
	FORMAT_ARGB8888 	  = 4,
	FORMAT_RGBA8888 	  = 5,
};

enum output_channel_e {
	OUTPUT_CHANNEL0 	  = 1,
	OUTPUT_CHANNEL1		  = 2,
	OUTPUT_CHANNEL2		  = 4,
	OUTPUT_CHANNEL3		  = 8,
};

enum output_mode_e {
	OUTPUT_MIPI 		  = 0,
	OUTPUT_BT1120		  = 1,
	OUTPUT_RGB888		  = 2,
};

struct disp_timing {
        uint32_t hbp;
        uint32_t hfp;
        uint32_t hs;
        uint32_t vbp;
        uint32_t vfp;
        uint32_t vs;
        uint32_t vfp_cnt;
};

typedef struct type_channel_base_cfg_t {
	uint32_t	channel;
	uint32_t	enable;
	uint32_t	pri;
	uint32_t	width;
	uint32_t	height;
	uint32_t	xposition;
	uint32_t	yposition;
	uint32_t format;
	uint32_t	alpha;
	uint32_t	keycolor;
	uint32_t	alpha_sel;
	uint32_t	ov_mode;
	uint32_t	alpha_en;
	uint32_t	crop_width;
	uint32_t	crop_height;
} channel_base_cfg_t;

typedef struct type_ppcon1_cfg_t {
	uint32_t	dithering_flag;
	uint32_t	dithering_en;
	uint32_t	gamma_en;
	uint32_t	hue_en;
	uint32_t	sat_en;
	uint32_t	con_en;
	uint32_t	bright_en;
	uint32_t	theta_sign;
	uint32_t	contrast;
} ppcon1_cfg_t;

typedef struct type_ppcon2_cfg_t {
	uint32_t	theta_abs; //ppcon2
	uint32_t	saturation;
	uint32_t	off_contrast;
	uint32_t	off_bright;
	float32_t	gamma_value;
} ppcon2_cfg_t;

typedef struct type_refresh_cfg_t {
	uint32_t	dbi_refresh_mode; //refresh mode
	uint32_t	panel_corlor_type;
	uint32_t	interlace_sel;
	uint32_t	odd_polarity;
	uint32_t	pixel_rate;
	uint32_t	ycbcr_out;
	uint32_t	uv_sequence;
	uint32_t	itu_r656_en;

	uint32_t	auto_dbi_refresh_cnt;
	uint32_t	auto_dbi_refresh_en;
} refresh_cfg_t;

typedef struct type_output_cfg_t {
	uint32_t	bgcolor;
	uint32_t	out_sel;
	uint32_t	width;
	uint32_t	height;
	uint32_t        big_endian;
	uint32_t	display_addr_type;
	uint32_t	display_cam_no;
	uint32_t        display_addr_type_layer1;
	uint32_t        display_cam_no_layer1;
	ppcon1_cfg_t	ppcon1;
	ppcon2_cfg_t	ppcon2;
	refresh_cfg_t	refresh_cfg;
	uint32_t	panel_type;
	uint32_t	rotate;
	uint32_t	user_control_disp;
	uint32_t	user_control_disp_layer1;
} output_cfg_t;

typedef struct type_upscaling_cfg_t {
	uint32_t	enable;
	uint32_t	src_width;
	uint32_t	src_height;
	uint32_t	tgt_width;
	uint32_t	tgt_height;
	uint32_t	step_x;
	uint32_t	step_y;
	uint32_t	pos_x;
	uint32_t	pos_y;
} upscaling_cfg_t;

#ifdef XJ3_WB
typedef struct _wb_cfg_t {
	uint32_t src_sel;
	uint32_t format;
} wb_cfg_t;
#endif

typedef struct type_frame_buf_t {
	void *vaddr;
	phys_addr_t paddr;
} frame_buf_t;

struct gamma_reg_bits_s {
	unsigned int part_a:8;
	unsigned int part_b:8;
	unsigned int part_c:8;
	unsigned int part_d:8;
};
typedef union type_gamma_para_t
{
	uint32_t	value;
	struct gamma_reg_bits_s bit;
}gama_para_t;

typedef struct type_gamma_cfg_t {
	gama_para_t    gamma_xr[4];
	gama_para_t    gamma_xg[4];
	gama_para_t    gamma_xb[4];
	gama_para_t    gamma_yr[4];
	gama_para_t    gamma_yg[4];
	gama_para_t    gamma_yb[4];
	gama_para_t    gamma_y16rgb;
} gamma_cfg_t;

typedef struct type_update_cmd_t
{
	uint32_t enable_flag[IDU_CHANNEL_MAX];
	uint32_t frame_size[IDU_CHANNEL_MAX];
	frame_buf_t srcframe[IDU_CHANNEL_MAX];
}update_cmd_t;

struct display_video_vaddr {
        void *channel0_y_addr;
        void *channel0_c_addr;
        void *channel1_y_addr;
        void *channel1_c_addr;
};

struct display_vio_channel_pipe {
        uint8_t disp_layer_no;
        uint8_t vio_pipeline;
        uint8_t vio_channel;
};

struct idu_handle_s {
	int32_t idudev_fd;
	int32_t lt9211_fd;
	int32_t channel_fd[IDU_CHANNEL_MAX];
	update_cmd_t update_cmd;
	uint32_t memory_mode;
	channel_base_cfg_t channel_base_cfg[IDU_CHANNEL_MAX];
	gamma_cfg_t gamma_cfg;
	output_cfg_t output_cfg;
	upscaling_cfg_t scale_cfg;
	struct disp_timing timing;
	char *framebuf[IDU_CHANNEL_MAX];
#ifdef XJ3_WB
	buffer_mgr_t *idu_output_bufmgr[IDU_CHANNEL_MAX];
	buffer_mgr_t *idu_wb_bufmgr;
	wb_cfg_t wb_cfg;
#endif
};

/**
 * init IDU module
 * @cfg_file: path of the config json file
 * @return value: 0:success;1:fail
 * used for XJ3 platform
 */
int32_t hb_disp_init_cfg(const char *cfg_file);

/**
 * init IDU module
 * @cfg_file: path of the config json file
 * @return value: 0:success;1:fail
 * used for J5 platform
 */
int32_t hb_disp_init_dev_cfg(uint32_t disp_id, const char *cfg_file);

/**
 * close IDU module
 * @return value: 0:success;1:fail
 * use after hb_disp_stop
 */
int32_t hb_disp_close(void);

/**
 * start IDU module
 * @return value: 0:success;1:fail
 * use after hb_disp_init_dev_cfg
 */
int32_t hb_disp_start(void);

/**
 * stop IDU module
 * @return value: 0:success;1:fail
 * use after hb_disp_start
 */
int32_t hb_disp_stop(void);

/**
 * enable IDU layer
 * @layer_number: layer number user want to enable
 * @return value: 0:success;1:fail
 */
int32_t hb_disp_layer_on(uint32_t layer_number);

/**
 * disable IDU layer
 * @layer_number: layer number user want to disable
 * @return value: 0:success;1:fail
 */
int32_t hb_disp_layer_off(uint32_t layer_number);

/**
 * user sets the address of the images to be displayed
 * @addr0_y: the y address of the image to be displayed in IDU layer 0
 * @addr0_c: the uv address of the image to be displayed in IDU layer 0
 * @addr1_y: the y address of the image to be displayed in IDU layer 1
 * @addr1_c: the uv address of the image to be displayed in IDU layer 1
 * @return value: 0:success;1:fail
 * addr0_y and addr1_y should not both null
 */
int32_t hb_disp_set_video_bufaddr(void *addr0_y, void *addr0_c,
                        void *addr1_y, void *addr1_c);

/**
 * user set layer configuration dynamically
 * @layer_no: IDU layer number
 * @width: width of the layer
 * @height: height of the layer
 * @x_pos: x position of the layer
 * @y_pos: y position of the layer
 * @return value: 0:success;1:fail
 */
int32_t hb_disp_set_layer_cfg(uint32_t layer_no,
                uint32_t width, uint32_t height,
                uint32_t x_pos, uint32_t y_pos);

/**
 * user power on display device
 * @return value: 0:success;1:fail
 * used for XJ3/XJ2 platform
 */
int32_t hb_disp_power_on(void);

/**
 * user power off display device
 * @return value: 0:success;1:fail
 * used for XJ3/XJ2 platform
 */
int32_t hb_disp_power_off(void);

/**
 * user set display timing parameter
 * @user_timing: the timing parameter user want to set
 * @return value: 0:success;1:fail
 */
int32_t hb_disp_set_timing(struct disp_timing *user_timing);

/**
 * user get gamma config
 * @gamma_val: input gamma value pointer, store result
 * @return value: 0:success;1:fail
 */
int32_t hb_disp_get_gamma_cfg(float32_t *gamma_val);

/**
 * user set gamma config
 * @gamma_user: gamma value user want to set
 * @return value: 0:success;1:fail
 */
int32_t hb_disp_set_gamma_cfg(float32_t gamma_user);

/**
 * user get output config
 * @cfg: the output config parameters struct that user gets
 * @return value: 0:success;1:fail
 */
int32_t hb_disp_get_output_cfg(output_cfg_t *cfg);

/**
 * user set output config
 * @cfg: the output config parameters user want to set
 * @return value: 0:success;1:fail
 */
int32_t hb_disp_set_output_cfg(const output_cfg_t *cfg);

/**
 * user get up scale config
 * @cfg: the up scale config parameters struct that user gets
 * @return value: 0:success;1:fail
 */
int32_t hb_disp_get_upscaling_cfg(upscaling_cfg_t *cfg);

/**
 * user set up scale config
 * @cfg: the up scale config parameters user want to set
 * @return value: 0:success;1:fail
 */
int32_t hb_disp_set_upscaling_cfg(const upscaling_cfg_t *cfg);

/**
 * user get channel config
 * @cfg: the channel config parameters struct that user gets
 * @chn: the layer number user want to get
 * @return value: 0:success;1:fail
 */
int32_t hb_disp_get_channel_cfg(uint32_t chn, channel_base_cfg_t *cfg);

/**
 * user set channel config
 * @cfg: the channle config parameters user want to set
 * @chn: the layer number user want to set
 * @return value: 0:success;1:fail
 */
int32_t hb_disp_set_channel_cfg(uint32_t chn, const channel_base_cfg_t *cfg);

/**
 * user config up-scale
 * @src_w: width of the source graphic
 * @src_h: height of the source graphic
 * @tag_w: width of the target graphic
 * @tag_h: height of the target graphic
 * @return value: 0:success;1:fail
 */
int32_t hb_disp_out_upscale(uint32_t src_w, uint32_t src_h,
		uint32_t tag_w, uint32_t tag_h);

/**
 * user get the display done flag
 * @return value: 0:not done;1:done
 */
int32_t hb_disp_get_display_done(void);

/**
 * user set display pipeline and channel number of vio module
 * @disp_layer: layer number of IDU
 * @pipeline: pipeline number of VIO module
 * @channel_no: channel number of VIO module
 * @return value: 0:not done;1:done
 */
int32_t hb_disp_set_vio_channel(uint8_t disp_layer, uint8_t pipeline,
		uint8_t channel_no);

/**
 * user check whether the graphic size matches the channel
 * @graphic_size: the graphic size
 * @disp_layer_no: layer number of IDU
 * @return value: 0:match; 1:not match
 */
int32_t hb_disp_check_video_bufaddr_valid(size_t graphic_size,
                uint32_t disp_layer_no);

#if (defined XJ3_WB)
int32_t hb_disp_wb_start(uint32_t sel, uint32_t format);
int32_t hb_disp_wb_stop();
int32_t hb_disp_release_screen_frame(hb_vio_buffer_t * idu_buffer);
int32_t hb_disp_get_screen_frame(hb_vio_buffer_t * idu_buffer);
int32_t hb_disp_wb_setcfg(uint32_t sel, uint32_t format);

int32_t hb_disp_output_start(uint32_t layer_no);
int32_t hb_disp_output_start_withoutbufmgr(uint32_t layer_no);
int32_t hb_disp_output_stop_withoutbufmgr(uint32_t layer_no);
int32_t hb_disp_send_frame(uint32_t layer_no, hb_vio_buffer_t *idu_buffer);
int32_t hb_disp_get_frame(uint32_t layer_no, hb_vio_buffer_t *idu_buffer);
int32_t idu_output_node_qbuf(int32_t fd, uint32_t layer_no,
		hb_vio_buffer_t * buf);
buf_node_t *idu_output_dqbuf_from_Reprocess(int32_t fd,
		uint32_t layer_no, buffer_mgr_t * buf_mgr);
int32_t idu_wb_node_qbuf(int32_t fd, hb_vio_buffer_t * buf);
buf_node_t *idu_wb_dqbuf(int32_t fd, buffer_mgr_t * buf_mgr);
#endif

#ifdef __cplusplus
}
#endif
#endif
