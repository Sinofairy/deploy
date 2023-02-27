/***************************************************************************
* COPYRIGHT NOTICE
* Copyright 2020 Horizon Robotics, Inc.
* All rights reserved.
***************************************************************************/
#ifndef J5_VIDEOSTITCH_INC_HB_VIDEOSTITCH_INTERFACE_H_
#define J5_VIDEOSTITCH_INC_HB_VIDEOSTITCH_INTERFACE_H_
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VIDEOSTITCH_BUFFER_MAX_PLANES (3u)
#define FRAME_MAX (4)
#define ROI_MAX (32u)
#define ROI_W_MAX (2000u)
#define ROI_H_MAX (2000u)
#define DST_W_MAX (4096u)
#define DST_H_MAX (2160u)


//#define HB_VPS_VIDEOSTITCH_NO_ERROR (0)
//#define HB_VPS_VIDEOSTITCH_FD_NOT_INITED (1)
#define	HB_VPS_VIDEOSTITCH_SET_HW_SETTINGS_FAILED (2)
//#define	HB_VPS_VIDEOSTITCH_WAIT_DONE_FAIL (3)
//#define	HB_VPS_VIDEOSTITCH_IOMMU_MAP_FAILED (4)
//#define	HB_VPS_VIDEOSTITCH_NOMEM (5)
#define	HB_VPS_VIDEOSTITCH_CFG_ATTR (6)
#define	HB_VPS_VIDEOSTITCH_WAITE_OUT (7)
#define	HB_VPS_VIDEOSTITCH_ADDRESS_ERR (8)

#define VIDEOSTITCH_ERR_BASE (8000u)

#define	HB_VPS_VIDEOSTITCH_NO_ERROR 0
#define	HB_VPS_VIDEOSTITCH_NOMEM (VIDEOSTITCH_ERR_BASE + 1)
#define	HB_VPS_VIDEOSTITCH_FD_NOT_INITED (VIDEOSTITCH_ERR_BASE + 2)
#define	HB_VPS_VIDEOSTITCH_INPUT_CGP_SIZE_ERR (VIDEOSTITCH_ERR_BASE + 3)
#define	HB_VPS_VIDEOSTITCH_INPUT_CALC_OUTBUF_ERR (VIDEOSTITCH_ERR_BASE + 4)
#define	HB_VPS_VIDEOSTITCH_ROI_NUM_ERR (VIDEOSTITCH_ERR_BASE + 5)
#define	HB_VPS_VIDEOSTITCH_CGP_NULL_ERR (VIDEOSTITCH_ERR_BASE + 6)
#define	HB_VPS_VIDEOSTITCH_CGP_ADDRESS_ERR (VIDEOSTITCH_ERR_BASE + 7)
#define	HB_VPS_VIDEOSTITCH_CGP_ADDRESS_SIZE_ERR (VIDEOSTITCH_ERR_BASE + 8)
#define	HB_VPS_VIDEOSTITCH_FRAME_NULL_ERR (VIDEOSTITCH_ERR_BASE + 9)
#define	HB_VPS_VIDEOSTITCH_FRAME_ADDRESS_ERR (VIDEOSTITCH_ERR_BASE + 10)
#define	HB_VPS_VIDEOSTITCH_FRAME_ADDRESS_SIZE_ERR (VIDEOSTITCH_ERR_BASE + 11)
#define	HB_VPS_VIDEOSTITCH_FRAME_FRAMENUM_ERR (VIDEOSTITCH_ERR_BASE + 12)
#define	HB_VPS_VIDEOSTITCH_FRAME_FRAMEREADY_ERR (VIDEOSTITCH_ERR_BASE + 13)
#define	HB_VPS_VIDEOSTITCH_FRAME_HEIGHT_WIDTH_ERR (VIDEOSTITCH_ERR_BASE + 14)
#define	HB_VPS_VIDEOSTITCH_FRAME_PICID_ERR (VIDEOSTITCH_ERR_BASE + 15)
#define	HB_VPS_VIDEOSTITCH_FRAME_STRIDE_ERR (VIDEOSTITCH_ERR_BASE + 16)
#define	HB_VPS_VIDEOSTITCH_RESULT_NULL_ERR (VIDEOSTITCH_ERR_BASE + 17)
#define	HB_VPS_VIDEOSTITCH_RESULT_ADDRESS_ERR (VIDEOSTITCH_ERR_BASE + 18)
#define	HB_VPS_VIDEOSTITCH_RESULT_ADDRESS_SIZE_ERR (VIDEOSTITCH_ERR_BASE + 19)
#define	HB_VPS_VIDEOSTITCH_RESULT_PLANE_ERR (VIDEOSTITCH_ERR_BASE + 20)
#define	HB_VPS_VIDEOSTITCH_LUT_NULL_ERR (VIDEOSTITCH_ERR_BASE + 21)
#define	HB_VPS_VIDEOSTITCH_LUT_ADDRESS_ERR (VIDEOSTITCH_ERR_BASE + 22)
#define	HB_VPS_VIDEOSTITCH_LUT_ADDRESS_SIZE_ERR (VIDEOSTITCH_ERR_BASE + 23)
#define	HB_VPS_VIDEOSTITCH_ROICFG_NULL_ERR (VIDEOSTITCH_ERR_BASE + 24)
#define	HB_VPS_VIDEOSTITCH_ROICFG_SRC_IDX_ERR (VIDEOSTITCH_ERR_BASE + 25)
#define	HB_VPS_VIDEOSTITCH_ROICFG_EQUAL_FRAME (VIDEOSTITCH_ERR_BASE + 26)
#define	HB_VPS_VIDEOSTITCH_ROICFG_SRC_BOUNDARY_ERR (VIDEOSTITCH_ERR_BASE + 27)
#define	HB_VPS_VIDEOSTITCH_ROICFG_DST_BOUNDARY_ERR (VIDEOSTITCH_ERR_BASE + 28)
#define	HB_VPS_VIDEOSTITCH_ROICFG_ROI_SIZE_ERR (VIDEOSTITCH_ERR_BASE + 29)
#define	HB_VPS_VIDEOSTITCH_ROICFG_MULTIPLES_OF_2_ERR (VIDEOSTITCH_ERR_BASE + 30)
#define	HB_VPS_VIDEOSTITCH_ROICFG_SRC_SIZE_ERR (VIDEOSTITCH_ERR_BASE + 31)
#define	HB_VPS_VIDEOSTITCH_ROICFG_DST_SIZE_ERR (VIDEOSTITCH_ERR_BASE + 32)
#define	HB_VPS_VIDEOSTITCH_ROICFG_MODE_ERR (VIDEOSTITCH_ERR_BASE + 33)
#define	HB_VPS_VIDEOSTITCH_ROICFG_DIRECT_ERR (VIDEOSTITCH_ERR_BASE + 34)
#define	HB_VPS_VIDEOSTITCH_ROICFG_MARGIN_ERR (VIDEOSTITCH_ERR_BASE + 35)
#define	HB_VPS_VIDEOSTITCH_ROICFG_MARGIN_INV_ERR (VIDEOSTITCH_ERR_BASE + 36)
#define	HB_VPS_VIDEOSTITCH_ROICFG_MODE_ONLINE_ERR (VIDEOSTITCH_ERR_BASE + 37)
#define	HB_VPS_VIDEOSTITCH_ROICFG_GAIN_ERR (VIDEOSTITCH_ERR_BASE + 38)
#define	HB_VPS_VIDEOSTITCH_HW_PROCESS_ERR (VIDEOSTITCH_ERR_BASE + 39)
#define	HB_VPS_VIDEOSTITCH_HW_CHECK_ADDRESS_ERR (VIDEOSTITCH_ERR_BASE + 40)
#define	HB_VPS_VIDEOSTITCH_HW_MAP_ERR (VIDEOSTITCH_ERR_BASE + 41)
#define	HB_VPS_VIDEOSTITCH_HW_UNMAP_ERR (VIDEOSTITCH_ERR_BASE + 42)
#define	HB_VPS_VIDEOSTITCH_HW_FRAME_Y_CHECK_ADDRESS_ERR (VIDEOSTITCH_ERR_BASE + 43)
#define	HB_VPS_VIDEOSTITCH_HW_FRAME_UV_CHECK_ADDRESS_ERR (VIDEOSTITCH_ERR_BASE + 44)
#define	HB_VPS_VIDEOSTITCH_HW_RESULT_CHECK_ADDRESS_ERR (VIDEOSTITCH_ERR_BASE + 45)
#define	HB_VPS_VIDEOSTITCH_HW_LUT_ALPHA_CHECK_ADDRESS_ERR (VIDEOSTITCH_ERR_BASE + 46)
#define	HB_VPS_VIDEOSTITCH_HW_LUT_BETA_CHECK_ADDRESS_ERR (VIDEOSTITCH_ERR_BASE + 47)
#define	HB_VPS_VIDEOSTITCH_HW_CGP_CHECK_ADDRESS_ERR (VIDEOSTITCH_ERR_BASE + 48)
#define	HB_VPS_VIDEOSTITCH_HW_FRAME_Y_MAP_ERR (VIDEOSTITCH_ERR_BASE + 49)
#define	HB_VPS_VIDEOSTITCH_HW_FRAME_UV_MAP_ERR (VIDEOSTITCH_ERR_BASE + 50)
#define	HB_VPS_VIDEOSTITCH_HW_RESULT_MAP_ERR (VIDEOSTITCH_ERR_BASE + 51)
#define	HB_VPS_VIDEOSTITCH_HW_LUT_ALPHA_MAP_ERR (VIDEOSTITCH_ERR_BASE + 52)
#define	HB_VPS_VIDEOSTITCH_HW_LUT_BETA_MAP_ERR (VIDEOSTITCH_ERR_BASE + 53)
#define	HB_VPS_VIDEOSTITCH_HW_CGP_MAP_ERR (VIDEOSTITCH_ERR_BASE + 54)
#define	HB_VPS_VIDEOSTITCH_HW_FRAME_Y_UNMAP_ERR (VIDEOSTITCH_ERR_BASE + 55)
#define	HB_VPS_VIDEOSTITCH_HW_FRAME_UV_UNMAP_ERR (VIDEOSTITCH_ERR_BASE + 56)
#define	HB_VPS_VIDEOSTITCH_HW_RESULT_UNMAP_ERR (VIDEOSTITCH_ERR_BASE + 57)
#define	HB_VPS_VIDEOSTITCH_HW_LUT_ALPHA_UNMAP_ERR (VIDEOSTITCH_ERR_BASE + 58)
#define	HB_VPS_VIDEOSTITCH_HW_LUT_BETA_UNMAP_ERR (VIDEOSTITCH_ERR_BASE + 59)
#define	HB_VPS_VIDEOSTITCH_HW_CGP_UNMAP_ERR (VIDEOSTITCH_ERR_BASE + 60)

enum {//enumeration of stitching mode
	MODE_ONLINE = 0, //online mode
	MODE_ALPHA = 1, //alpha mode
	MODE_ALPHA_BETA = 2, //alpha beda mode
	MODE_SRC = 3, //src mode
	MODE_ALPHA_SRC0 = 5 //arpha src0 mode
};

enum {//enumeration of frame index
	PIC_ID_LEFT = 0, //left
	PIC_ID_TOP, //top
	PIC_ID_RIGHT, //right
	PIC_ID_BOTTOM, //bottom
	PIC_ID_ALL, //all frame
	PIC_ID_MAX //max value
};

enum {//enumeration of direct
	DIRECT_LT = 0, //left and top direct
	DIRECT_RB = 1, //right and bottom direct
	DIRECT_LB = 2, //left and bottom direct
	DIRECT_RT = 3, //right and top direct
	DIRECT_MAX = 4 //max value of direct
};

//structure
typedef struct buffer_offset_s {
	uint32_t width; //width of dst image
	uint32_t height; //height of dst image
	uint32_t stride_size; //stride of dst image
	uint32_t size[VIDEOSTITCH_BUFFER_MAX_PLANES]; //size of dst image
	uint32_t planeCount; //plane count of dst image
	uint32_t offset[VIDEOSTITCH_BUFFER_MAX_PLANES]; //y,uv  or raw/ dol2 / dol3
}buffer_offset_t;

typedef struct src_rect_s {
	uint32_t	src0_idx; //index of frame which src0 belongs to

	uint32_t	src0_x; //initial x coordinate of src0
	uint32_t	src0_y; //initial y coordinate of src0

	uint32_t	src0_w; //width of src0
	uint32_t	src0_h; //height of src0

	uint32_t	src1_idx; //index of frame which src1 belongs to

	uint32_t	src1_x; //initial x coordinate of src1
	uint32_t	src1_y; //initial y coordinate of src1

	uint32_t	src1_w; //width of src1
	uint32_t	src1_h; //height of src1
}src_rect_t;

typedef struct dst_rect_s {
	uint32_t	dst_xpos; //initial x coordinate of roi in dst image
	uint32_t	dst_ypos; //initial y coordinate of roi in dst image

	uint32_t	roi_w; //width of roi
	uint32_t	roi_h; //height of roi

	uint32_t	dst_w; //width of dst image
	uint32_t	dst_h; //height of dst image
}dst_rect_t;

typedef struct ctrl_s {
	uint32_t	mode; //stitching mode
	uint32_t	direct; //online calculation mode direct
	uint32_t	uven; //dst image plane
	uint32_t	margin; //margin attribute ,the default value is 10
	uint32_t	margin_inv; //when margin>0, floor(128*128/margin_inv).when margin==0,margin_inv==128
}ctrl_t;

typedef struct src_gain_s {
	uint32_t	src0_gain_y; //gain of y src0
	uint32_t	src1_gain_y; //gain of y src1

	uint32_t	src0_gain_u; //gain of u src0
	uint32_t	src1_gain_u; //gain of u src1

	uint32_t	src0_gain_v; //gain of v src0
	uint32_t	src1_gain_v; //gain of v src1
}src_gain_t;

typedef struct roi_cfg_s {
	uint32_t	index; //index of roi
	src_rect_t	src_rect; //src_rect_t data structure
	dst_rect_t	dst_rect; //dst_rect_t data structure
	ctrl_t 		ctrl; //ctrl_t data structure
	src_gain_t 	src_gain; //src_gain_t data structure
}roi_cfg_t;

typedef struct result_addr_s {
	uint32_t 	planeCount; //plane count of dst image
	uint32_t		size[VIDEOSTITCH_BUFFER_MAX_PLANES]; //size of dst image
	char		*virt_img[VIDEOSTITCH_BUFFER_MAX_PLANES]; //virtual addrass of dst image
	uint64_t		pa_img[VIDEOSTITCH_BUFFER_MAX_PLANES]; //physical addrass of dst image
}result_addr_t;

typedef struct cgp_addr_s {
	uint32_t	size; //size of config package
	char		*virt_cgp; //virtual addrass of config package
	uint64_t	pa_cgp; //physical addrass of config package
}cgp_addr_t;

typedef struct lut_table_addr_s {
	uint32_t          size; //size of look-up-table
	char              *lut_parameter; //virtual addrass of look-up-table
	uint64_t          pa_lut; //physical addrass of look-up-table
}lut_table_addr_t;

typedef struct lut_table_s {
	lut_table_addr_t	alpha_lut_table; //alpha lut_table_addr_s data structure
	lut_table_addr_t	beta_lut_table; //beta lut_table_addr_s data structure
}lut_table_t;

typedef struct frame_s {
	uint32_t            pic_id; //frame id
	uint32_t            height; //height of frame
	uint32_t            width; //width of frame
	uint32_t            y_stride; //y_stride of frame
	uint32_t            uv_stride; //uv_stride of frame
	uint64_t            phy_y_addr; //physical y addrass of frame
	uint64_t            phy_uv_addr; //physical uv addrass of frame
	char                *virt_y_addr; //virtual y addrass of frame
	char                *virt_uv_addr; //virtual uv addrass of frame
	uint32_t            y_size; //y size of frame
    uint32_t            uv_size; //uv size of frame
}frame_t;

typedef struct frameinfo_s {
	uint32_t	frame_num;
	//PIC_ID_LEFT-L,PIC_ID_TOP-LT,PIC_ID_RIGHT-LTR,PIC_ID_BOTTOM-LTRB,PIC_ID_ALL-ALL
	int32_t		frame_ready; //ready count of frame
	frame_t		frame[FRAME_MAX];//0-left,1-top,2-right,3-bottom
}frameinfo_t;
/* info :
 * y,uv             2 plane
 **/

/**
 * hb_videostitch_init() - request and initialize module resource.
 * Return:0 success, or error code
 */
int32_t hb_videostitch_init(void);

/**
 * hb_videostitch_deinit() - free requested resource.
 * Return:0 success, or error code
 */
int32_t hb_videostitch_deinit(void);

/**
 * hb_videostitch_calc_outbuf_size() - compute output buffer size.
 * @width: the width of output buffer.
 * @height: the height of output buffer.
 * @buf_offset: the information of the output buffer.
 * Return:0 success, or error code
 */
int32_t hb_videostitch_calc_outbuf_size(uint32_t width, uint32_t height, buffer_offset_t *buf_offset);

/**
 * hb_videostitch_cgp_size() - compute configuration word size.
 * @roi_num: roi number.
 * @size: configuration word size.
 * Return:0 success, or error code
 */
int32_t hb_videostitch_cgp_size(uint32_t roi_num, uint32_t *size);

/**
 * hb_videostitch_release() - start once stitching and get the stitched image(output buffer).
 * @roi_cfg: roi configuration
 * @frameinfo: input buffer information
 * @result_addr: output buffer address
 * @cgp_addr: configuration word address
 * @lut_table: LUT table address
 * Return:0 success, or error code
 */
int32_t hb_videostitch_process(uint32_t		roi_num,
				const roi_cfg_t	*roi_cfg,
				const frameinfo_t	*frameinfo,
				const result_addr_t	*result_addr,
				const cgp_addr_t	*cgp_addr,
				const lut_table_t	*lut_table);

#ifdef __cplusplus
	}
#endif


#endif //J5_VIDEOSTITCH_INC_HB_VIDEOSTITCH_INTERFACE_H_
