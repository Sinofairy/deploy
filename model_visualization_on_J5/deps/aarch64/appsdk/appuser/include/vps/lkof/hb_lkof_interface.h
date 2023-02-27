/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2020 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/
#ifndef HB_LKOF_INTERFACE_H
#define HB_LKOF_INTERFACE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define phy_addr_t uint64_t
#define ROI_MAX (4095u)
#define LAYER0_W_MAX (1920u)
#define LAYER0_H_MAX (1080u)
#define LAYER0_STRIDE_MAX (4096u)

#define LKOF_CHECK_POINTS	"LKOF_CHECK_POINTS"

#define CHIP_BUG_PRY_START (2u)
#define CHIP_BUG_DST_H     (64u)
#define CHIP_BUG_DST_W     (1024u)
#define CHIP_BUG_NPOINTS   (1000u)
#define CHIP_BUG_BLOCK_SIZE_W   (480u)
#define CHIP_BUG_BLOCK_SIZE_H   (64u)


#define MAGIC_NUM0_VAL (19951111u)
#define MAGIC_NUM1_VAL (19881102u)
#define MAGIC_NUM2_VAL (19851106u)
#define MAGIC_NUM3_VAL (20191205u)
#define MAGIC_NUM4_VAL (20200518u)

#define FC_HEAD_BYTES (64u)
#define MAGIC_BYTES (16u)
#define OPT_PTS_BITS (3u)
#define HARRIS_PTS_BITS (6u)
#define CONF_PTS_BITS (6u)
#define RESULT_OUTPUT_BYTES (16u)
#define ROI_POINTS_NUM (3u)

#define LAST_ROI_NO (0u)
#define LAST_ROI_YES (1u)

//ERROR CODE
#define LKOF_ERR_BASE  (9000u)
#define	HB_VPS_LKOF_OK (0)
#define	HB_VPS_LKOF_FD_NOT_INITED (LKOF_ERR_BASE + 1)
#define	HB_VPS_LKOF_HW_FAILED (LKOF_ERR_BASE + 2)
#define	HB_VPS_LKOF_TIMEOUT (LKOF_ERR_BASE + 3)
#define	HB_VPS_LKOF_NODATA (LKOF_ERR_BASE + 4)
#define	HB_VPS_LKOF_INVAL (LKOF_ERR_BASE + 5)
#define	HB_VPS_LKOF_CACHE_FLUSH_ERR (LKOF_ERR_BASE + 6)
#define	HB_VPS_LKOF_CACHE_CHECK_ERR (LKOF_ERR_BASE + 7)
#define	HB_VPS_LKOF_FRAMEID_NOT_UNIQUE (LKOF_ERR_BASE + 8)
#define HB_VPS_LKOF_NULL_POINTER (LKOF_ERR_BASE + 9)
#define	HB_VPS_LKOF_NOMEM (LKOF_ERR_BASE + 10)
#define	HB_VPS_LKOF_FIFO_OVERFLOW (LKOF_ERR_BASE + 11)
#define	HB_VPS_LKOF_OVERFLOW (LKOF_ERR_BASE + 12)


typedef enum {
	MODE_PROCESS_TYPE_ASYNCH,//0
	MODE_PROCESS_TYPE_SYNCH,
	MODE_PROCESS_TYPE_MAX,
}lkof_process_type;

typedef enum {
	INITIAL_MODE_NO = 0,
	INITIAL_MODE_YES = 1,
	INITIAL_MODE_MAX,
}lkof_initial_mode;

typedef enum {
	SRC_MODE_SAMPLE_NO = 0,
	SRC_MODE_SAMPLE_YES = 1,
	SRC_MODE_MAX,
}lkof_src_mode;

typedef enum {
	RESULT_MEM_TYPE_OPT,
	RESULT_MEM_TYPE_HARRIS,
	RESULT_MEM_TYPE_CONF,
	RESULT_MEM_TYPE_MAX,
}result_mem_type_t;

typedef struct point_info_s {
	uint8_t		status;
	uint32_t	point_y;
	uint32_t	point_x;
}point_info_t;

typedef struct src_sample_param_s {
	uint32_t	src_left;
	uint32_t	src_top;
	uint32_t	src_w;
	uint32_t	src_h;
	uint32_t	src_step_w;
	uint32_t	src_step_h;
}src_sample_param_t;

typedef struct dst_sample_param_s {
	uint32_t	dst_init_step_w;
	uint32_t	dst_init_step_h;
	uint32_t	dst_init_x;
	uint32_t	dst_init_y;
}dst_sample_param_t;

typedef struct harris_s {
	uint32_t	filter_en;
	uint32_t	filter_k;
	uint32_t	filter_threshold;
	uint32_t	filter_win_size;
	uint32_t	filter_out_en;
}harris_t;

typedef struct opt_ctrl_s {
	uint32_t	opt_en;
	uint32_t	initial_mode;
	uint32_t	src_mode;
	uint32_t	max_iteration;
	uint32_t	border;
	uint32_t	n_points;
	uint32_t	pry_end;
	uint32_t	pry_start;
	uint32_t	flow_win_size;
	uint32_t	conf_en;
	uint32_t	epsinon;
	struct point_info_s	*srcpt;
	struct point_info_s	*initpt;
}opt_ctrl_t;

typedef struct dst_roi_s {
	uint32_t	dst_left;
	uint32_t	dst_top;
	uint32_t	dst_w;
	uint32_t	dst_h;
}dst_roi_t;

typedef struct block_parameter_s {
	uint32_t	block_size_w;
	uint32_t	block_size_h;
}block_parameter_t;

typedef struct result_addr_s {
	char			*virt_lkof;
	char			*virt_harris;
	char			*virt_confidence;
	uint64_t		pa_lkof;
	uint64_t		pa_harris;
	uint64_t		pa_confidence;
	uint32_t 		lkof_size;
	uint32_t 		harris_size;
	uint32_t 		confidence_size;
}result_addr_t;

typedef struct fc_addr_s {
	uint32_t		size;
	char			*virt_fc;
	uint64_t		pa_fc;
}fc_addr_t;

typedef struct pym_layer_info_s {
	uint32_t	pre_layer_ddr_offset[4];
	uint32_t	next_layer_ddr_offset[4];
	uint32_t	layer_w[5];
	uint32_t	layer_h[5];
	uint32_t	layer_stride[5];
}pym_layer_info_t;

typedef struct image_address_s {
	uint64_t	src_image_address_layer0;
	uint32_t 	srcimg_size;
	uint64_t	dst_image_address_layer0;
	uint32_t 	dstimg_size;
	struct pym_layer_info_s pym_layer_info;
}image_address_t;

typedef struct one_roi_param_s {
	src_sample_param_t	src_sample_param;
	dst_sample_param_t	dst_sample_param;
	dst_roi_t		dst_roi;
	block_parameter_t	block_parameter;
	harris_t		harris;
	opt_ctrl_t		opt_ctrl;
}one_roi_param_t;

typedef struct kfifo_info_s {
	uint32_t		cap;
	uint32_t		current_num;
}kfifo_info_t;

typedef struct fifo_info_status_s {
	kfifo_info_t input_kfifo_info;
	kfifo_info_t output_kfifo_info;
}fifo_info_status_t;

/**
 * hb_lkof_init() - request and initialize module resource.
 * Return:0 success, or error code
 */
int32_t hb_lkof_init(void);

/**
 * hb_lkof_deinit() - free requested resource.
 * Return:0 success, or error code
 */
int32_t hb_lkof_deinit(void);

/**
 * hb_lkof_ftoi() - convert floating point to fixed point.
 * @point: floating point.
 * @out: fixed point.
 * Return:0 success, or error code
 */
int32_t hb_lkof_ftoi(float point, uint32_t *out);/*PRQA S ALL*/

/**
 * hb_lkof_itof() - convert fixed point to floating point.
 * @value: fixed point.
 * @result_mem_type: hardware support result type, contain harris/optical flow/confidence.
 * @out: floating point
 * Return:0 success, or error code
 */
int32_t hb_lkof_itof(int32_t value, result_mem_type_t result_mem_type, float *out);/*PRQA S ALL*/

/**
 * hb_lkof_trackpoint() - setting compute paramters and get result when use synchronization mode.
 * @image_address: input gaussian pyramid layer information.
 * @roi_num_in: roi numbers.
 * @src_sample_param: src sample paramter.
 * @dst_sample_param: dst sample paramter.
 * @dst_roi: dst roi step.
 * @block_parameter: block size of width and height.
 * @harris: harris ctrl paramter.
 * @opt_ctrl: optical flow ctrl paramter
 * @fc_addr: function call address information.
 * @mode: synchronous/asynchronous mode.
 * @frame_id: unique frame id bind with computing tasks.
 * @user_err: the status of current computing task.
 * @result_addr: output result address.
 * Return:0 success, or error code
 */
int32_t hb_lkof_trackpoint(image_address_t			image_address,
					uint32_t			roi_num_in,
					src_sample_param_t		const *src_sample_param,
					dst_sample_param_t		const *dst_sample_param,
					dst_roi_t			const *dst_roi,
					block_parameter_t		const *block_parameter,
					harris_t				const *harris,
					opt_ctrl_t			const *opt_ctrl,
					fc_addr_t			const *fc_addr,
					lkof_process_type		mode,
					uint64_t				frame_id,
					result_addr_t			*result_addr,
					int32_t 				*user_err);

/**
 * hb_lkof_trackpoint_params_one_struct() - setting compute paramters and get result when use synchronization mode,
   Combining parameters in a structure.
 * @image_address: input gaussian pyramid layer information.
 * @roi_num_in: roi numbers.
 * @one_roi_param: all one roi required ctrl paramters.
 * @fc_addr: function call address information.
 * @mode: synchronous/asynchronous mode.
 * @frame_id: unique frame id bind with computing tasks.
 * @user_err: the status of current computing task.
 * @result_addr: output result address.
 * Return:0 success, or error code
 */
int32_t hb_lkof_trackpoint_params_one_struct(image_address_t			image_address,
						uint32_t		roi_num_in,
						one_roi_param_t		const *one_roi_param,
						fc_addr_t		const *fc_addr,
						lkof_process_type	mode,
						uint64_t		frame_id,
						result_addr_t		*result_addr,
						int32_t 		*user_err);


/**
 * hb_lkof_wait_done() - get the result when use asynchronization mode.
 * @mode: synchronous/asynchronous mode, now only support asynchronous mode.
 * @frame_id: set the frame id of the current computing task.
 * @user_err: the status of current computing task.
 * @timeout: the wait time of getting current computing task.
 * Return:0 success, or error code
 */
int32_t hb_lkof_wait_done(lkof_process_type mode, uint64_t frame_id, int32_t *user_err, int32_t timeout);

/**
 * hb_lkof_compute_rms() - compute the memory size of one roi for result buffer.
 * @roi_num_in: roi number.
 * @n_points: the point number of one roi.
 * @size: the size of result buffer.
 * @result_mem_type: hardware support result type, contain harris/optical flow/confidence.
 * @last_roi: when current roi is last roi, memory size need plus magic size(16 bytes).
 * Return:0 success, or error code
 */
int32_t hb_lkof_compute_rms(uint32_t roi_num_in, uint32_t const *n_points, uint32_t *size, result_mem_type_t result_mem_type, uint32_t last_roi);

/**
 * hb_lkof_compute_fms() - compute the memory size of one roi for result buffer.
 * @roi_num_in: roi number.
 * @opt_ctrl: optical flow ctrl paramter
 * @size: the size of function call buffer.
 * Return:0 success, or error code
 */
int32_t hb_lkof_compute_fms(uint32_t roi_num_in, opt_ctrl_t const *opt_ctrl, uint32_t *size);

/**
 * hb_lkof_get_fifo_status() - get input and output kfifo information status.
 * @fifo_info_status_t: input and output kfifo status.
 * Return:0 success, or error code
 */
int32_t hb_lkof_get_fifo_status(fifo_info_status_t *fifo_status);
#ifdef __cplusplus
	}
#endif
#endif //HB_LKOF_INTERFACE_H
