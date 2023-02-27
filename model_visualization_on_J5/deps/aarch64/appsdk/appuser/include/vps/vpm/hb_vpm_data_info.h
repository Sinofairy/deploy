/***************************************************************************
* COPYRIGHT NOTICE
* Copyright 2019 Horizon Robotics, Inc.
* All rights reserved.
***************************************************************************/
#ifndef J5_VPM_INC_HB_VPM_DATA_INFO_H
#define J5_VPM_INC_HB_VPM_DATA_INFO_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include <sys/time.h>

#define HB_VIO_PIPELINE_MAX 16u
#define HB_VIO_PYM_MAX_BASE_LAYER 6u
#define HB_VIO_BUFFER_MAX_PLANES 3u
#define HB_VIO_BUFFER_LAST_PLANES 2
#define HB_VIO_CIM_BUFF_MAX_NUM   (32)
#define HB_VIO_ISP_YUV_BUFF_MAX_NUM (32)
#define HB_VIO_ISP_RAW_BUFF_MAX_NUM (32)
#define HB_VIO_PYM_BUFF_MAX_NUM   (64)
#define HB_VIO_PYM_FB_BUFF_MAX_NUM  (32)
#define HB_VIO_GDC_FB_BUFF_MAX_NUM  (32)
#define HB_VIO_GDC_BUFF_MAX_NUM  (32)

// callback enable when bit set 1 else disable the cb
#define HB_VIO_IPU_DS0_CB_MSG	0x001
#define HB_VIO_IPU_DS1_CB_MSG	0x002
#define HB_VIO_IPU_DS2_CB_MSG	0x004
#define HB_VIO_IPU_DS3_CB_MSG	0x008
#define HB_VIO_IPU_DS4_CB_MSG	0x010
#define HB_VIO_IPU_US_CB_MSG	0x020
#define HB_VIO_PYM_CB_MSG		0x040
#define HB_VIO_DIS_CB_MSG		0x080
#define HB_VIO_PYM_VER2_CB_MSG		0x100

#define HB_VIO_PYM_MAX_LAYER      18u
#define HB_VIO_PYM_LAYER_Index_2  2u
#define HB_VIO_PYM_LAYER_Index_5  5u
#define HB_VIO_PYM_LAYER_Index_8  8u
#define HB_VIO_PYM_LAYER_Index_13 13u

#define HW_FORMAT_YUV422_8BIT	0x1E
#define HW_FORMAT_YUV422_10BIT	0x1F
#define HW_FORMAT_RAW8		0x2A
#define HW_FORMAT_RAW10 	0x2B
#define HW_FORMAT_RAW12 	0x2C
#define HW_FORMAT_RAW14 	0x2D
#define HW_FORMAT_RAW16 	0x2E
#define HW_FORMAT_RAW20 	0x2F

#define GDC_FB_BUFFER_MASK 14u
#define GDC_DATA_BUFFER_MASK 12u

typedef enum VIO_INFO_TYPE_S {
	HB_VIO_CALLBACK_ENABLE = 0,
	HB_VIO_CALLBACK_DISABLE,
	HB_VIO_IPU_SIZE_INFO,	//reserve for ipu size setting in dis
	HB_VIO_IPU_US_IMG_INFO,
	HB_VIO_IPU_DS0_IMG_INFO,
	HB_VIO_IPU_DS1_IMG_INFO,
	HB_VIO_IPU_DS2_IMG_INFO,
	HB_VIO_IPU_DS3_IMG_INFO,
	HB_VIO_IPU_DS4_IMG_INFO,
	HB_VIO_PYM_IMG_INFO,
	HB_VIO_ISP_IMG_INFO,
	HB_VIO_PYM_V2_IMG_INFO,
	HB_VIO_INFO_MAX
} VIO_INFO_TYPE_E;

typedef enum VIO_CALLBACK_TYPE {
	HB_VIO_IPU_DS0_CALLBACK = 0,
	HB_VIO_IPU_DS1_CALLBACK,
	HB_VIO_IPU_DS2_CALLBACK,
	HB_VIO_IPU_DS3_CALLBACK,
	HB_VIO_IPU_DS4_CALLBACK,
	HB_VIO_IPU_US_CALLBACK,
	HB_VIO_PYM_CALLBACK,	// 6
	HB_VIO_DIS_CALLBACK,
	HB_VIO_PYM_V2_CALLBACK,
	HB_VIO_MAX_CALLBACK
} VIO_CALLBACK_TYPE_E;

typedef enum VIO_DATA_TYPE_S {
	HB_VIO_DATA_TYPE_INVALID = -1,
	HB_VIO_IPU_DS0_DATA = 0,
	HB_VIO_IPU_DS1_DATA,
	HB_VIO_IPU_DS2_DATA,
	HB_VIO_IPU_DS3_DATA,
	HB_VIO_IPU_DS4_DATA,
	HB_VIO_IPU_US_DATA,	// 5
	HB_VIO_PYM_FEEDBACK_SRC_DATA,	// for debug
	HB_VIO_PYM_DATA,//7
	HB_VIO_SIF_FEEDBACK_SRC_DATA,
	HB_VIO_SIF_RAW_DATA,
	HB_VIO_SIF_YUV_DATA,//10
	HB_VIO_ISP_YUV_DATA,	// for debug, a process result for raw feedback
	HB_VIO_GDC_DATA,
	HB_VIO_GDC1_DATA,
	HB_VIO_IARWB_DATA,
	HB_VIO_GDC_FEEDBACK_SRC_DATA,//15
	HB_VIO_GDC1_FEEDBACK_SRC_DATA,
	HB_VIO_PYM_LAYER_DATA,
	HB_VIO_MD_DATA,
	HB_VIO_ISP_RAW_DATA,
	HB_VIO_PYM_COMMON_DATA,
	HB_VIO_PYM_DATA_V2,//21
	HB_VIO_CIM_RAW_DATA,
	HB_VIO_CIM_YUV_DATA,
	HB_VIO_EMBED_DATA,
	HB_VIO_DATA_TYPE_MAX
} VIO_DATA_TYPE_E;

typedef enum buffer_state {
	BUFFER_AVAILABLE,
	BUFFER_PROCESS,
	BUFFER_DONE,
	BUFFER_REPROCESS,
	BUFFER_USER,
	BUFFER_INVALID
} buffer_state_e;

/* info :
 * y,uv             2 plane
 * raw              1 plane
 * raw, raw         2 plane(dol2)
 * raw, raw, raw    3 plane(dol3)
 **/
typedef struct address_info_s {
	uint16_t width;
	uint16_t height;
	uint16_t stride_size;
	char *addr[HB_VIO_BUFFER_MAX_PLANES];
	uint64_t paddr[HB_VIO_BUFFER_MAX_PLANES];
} address_info_t;

typedef struct image_info_s {
	uint16_t sensor_id;
	uint32_t pipeline_id;
	uint32_t frame_id;
	uint64_t time_stamp;//HW time stamp
	struct timeval tv;//system time of hal get buf
	int32_t buf_index;
	int32_t img_format;
	int32_t fd[HB_VIO_BUFFER_MAX_PLANES];//ion buf fd
	uint32_t size[HB_VIO_BUFFER_MAX_PLANES];
	uint32_t planeCount;
	uint32_t dynamic_flag;
	uint32_t water_mark_line;
	VIO_DATA_TYPE_E data_type;
	buffer_state_e state;
	uint64_t		desc;//temp desc for isp raw feedback. TODO:need remove
} image_info_t;

/*
 * buf type  fd[num]                        size[num]                   addr[num]
 *
 * sif buf : fd[0(raw)]                     size[0(raw)]                addr[0(raw)]
 * sif dol2: fd[0(raw),1(raw)]              size[0(raw),1(raw)]         addr[0(raw),1(raw)]
 * sif dol3: fd[0(raw),1(raw),2(raw)]       size[0(raw),1(raw),2(raw)]  addr[0(raw),1(raw),2(raw)]
 * ipu buf : fd[0(y),1(c)]                  size[0(y),1(c)]             addr[0(y),1(c)]
 * pym buf : fd[0(all channel)]             size[0(all output)]         addr[0(y),1(c)] * 24
 * */

typedef struct hb_vio_buffer_s {
	image_info_t img_info;
	address_info_t img_addr;
} hb_vio_buffer_t;

//use in j3
typedef struct pym_buffer_s {
	image_info_t pym_img_info;
	address_info_t pym[6];
	address_info_t pym_roi[6][3];
	address_info_t us[6];
	char *addr_whole[HB_VIO_BUFFER_MAX_PLANES];
	uint64_t paddr_whole[HB_VIO_BUFFER_MAX_PLANES];
	uint32_t layer_size[30][HB_VIO_BUFFER_MAX_PLANES];
} pym_buffer_t;//Keep xj3 data struct

//use in j3 j5
typedef struct pym_buffer_common_s {
	image_info_t pym_img_info;
	address_info_t pym[HB_VIO_PYM_MAX_BASE_LAYER];//only for base layer
} pym_buffer_common_t;

//use in j5
typedef struct pym_buffer_v2_s {
	image_info_t pym_img_info;
	address_info_t src_out;
	address_info_t us_roi;
	address_info_t ds_roi[6];
	address_info_t gs[5];
	address_info_t bl[5];
} pym_buffer_v2_t;

typedef struct chn_img_info_s {
	uint16_t width;
	uint16_t height;
	uint16_t format;
	uint16_t buf_count;
} chn_img_info_t;


typedef void (*data_cb) (uint32_t pipe_id, uint32_t event, void *data, void *userdata);

enum Format {
	HB_RGB,
	HB_YUV422,
	HB_YUV420SP,
	HB_YUV444,
	HB_RAW,//raw8
	HB_RAW10,
	HB_RAW12,
	HB_RAW16,
	HB_RAW24,
	HB_YUV420SP_RAW12, /* yuv420 & raw12 */
	HB_YUV422_RAW12, /* yuv422 & raw12 */
	HB_YUV420SP_RAW16, /* yuv420 & raw16 */
	HB_YUV422_RAW16, /* yuv422 & raw16 */
	HB_IR8,
	HB_DECOMP_RAW,//decomp 24bit
	HB_FORMAT_MAX
};

enum rotation_e {
	ROTATION_0 = 0,
	ROTATION_90 = 90,
	ROTATION_180 = 180,
	ROTATION_270 = 270
};

/**
 * GDC Frame format
 */
typedef enum {
    FMT_UNKNOWN,
    FMT_LUMINANCE,
    FMT_PLANAR_444,
    FMT_PLANAR_420,
    FMT_SEMIPLANAR_420,
    FMT_GDC_MAX
} frame_format_t;
//---------------------------------------------------------
/**
 * GDC Rectangle definition
 */
typedef struct {
    int32_t x; ///< Start x coordinate
    int32_t y; ///< Start y coordinate
    int32_t w; ///< width
    int32_t h; ///< height
} rect_t;
//---------------------------------------------------------
/**
 * GDC Types of supported transformations
 */
typedef enum {
    PANORAMIC,
    CYLINDRICAL,
    STEREOGRAPHIC,
    UNIVERSAL,
    CUSTOM,
    AFFINE,
    DEWARP_KEYSTONE
} transformation_t;
//---------------------------------------------------------

//---------------------------------------------------------
/**
 * GDC Resolution definition
 */
typedef struct {
    uint32_t w;  ///< width in pixels
    uint32_t h;  ///< height in pixels
} resolution_t;
//---------------------------------------------------------
/**
 * Grid point
 */
typedef struct {
    double x;  /*PRQA S 5209*/  ///< x coordinate
    double y;  /*PRQA S 5209*/ ///< y coordinate
} point_t;

//---------------------------------------------------------
/**
 * Custom transformation structure
 */
typedef struct {
    uint8_t full_tile_calc;
    uint16_t tile_incr_x;
    uint16_t tile_incr_y;
    int32_t w;          ///< Number or points in horizontal direction in the custom transformation grid
    int32_t h;          ///< Number or points in vertical direction in the custom transformation grid
    double centerx;     /*PRQA S 5209*/ ///< center along x axis
    double centery;     /*PRQA S 5209*/ ///< center along y axis
    point_t* points;    ///< array with points, number of elements = w*h
} custom_tranformation_t;


//---------------------------------------------------------
/**
 * Window definition and transformation parameters
 * For parameters meaning read GDC guide
 */
typedef struct {
    rect_t out_r;                   ///< Output window position and size
    transformation_t transform;     ///< Used transformation
    rect_t input_roi_r;
    int32_t pan;                    ///< Target shift in horizontal direction from centre of the output image in pixels
    int32_t tilt;                   ///< Target shift in vertical direction from centre of the output image in pixels
    double zoom;                    /*PRQA S 5209*/
					///< Target zoom dimensionless coefficient (must not be bigger than zero)
    double strength;                /*PRQA S 5209*/
					///< Dimensionless non-negative parameter defining the strength of transformation along X axis
    double strengthY;               /*PRQA S 5209*/
					///< Dimensionless non-negative parameter defining the strength of transformation along X axis
    double angle;                   /*PRQA S 5209*/ ///< Angle of main projection axis rotation around itself in degrees
    // universal transformation
    double elevation;               /*PRQA S 5209*/
					///< Angle in degrees which specify the main projection axis
    double azimuth;                 /*PRQA S 5209*/
					///< Angle in degrees which specify the main projection axis,
					//counted clockwise from North direction (positive to East)
    int32_t keep_ratio;             ///< Keep the same stretching strength in both horizontal and vertical directions
    double FOV_h;                   /*PRQA S 5209*/ ///< Size of output field of view in vertical dimension in degrees
    double FOV_w;                   /*PRQA S 5209*/ ///< Size of output field of view in horizontal dimension in degrees
    double cylindricity_y;          /*PRQA S 5209*/
					///< Level of cylindricity for target projection shape in vertical direction
    double cylindricity_x;          /*PRQA S 5209*/
					///< Level of cylindricity for target projection shape in horizontal direction
    char custom_file[128];          ///< File name of the file containing custom transformation description
    custom_tranformation_t custom;  ///< Parsed custom transformation structure
    double trapezoid_left_angle;    /*PRQA S 5209*/ ///< Left Acute angle in degrees between trapezoid base and leg
    double trapezoid_right_angle;   /*PRQA S 5209*/ ///< Right Acute angle in degrees between trapezoid base and leg
} window_t;
//---------------------------------------------------------
/**
 * Common parameters
 */
typedef struct {
    frame_format_t format;  //frame format
    resolution_t in;        //input frame resolution
    resolution_t out;       //output frame resolution
    int32_t x_offset;       //center offset for input x coordinate
    int32_t y_offset;       //center offset for input y coordinate
    int32_t diameter;       //diameter of equivalent 180 field of view in pixels
    double fov;             /*PRQA S 5209*/ //field of view
} param_t;

typedef struct gdc_config {
	uint64_t config_addr;	//gdc config address
	uint32_t config_size;	//gdc config size in 32bit
	uint32_t input_width;	//gdc input width resolution
	uint32_t input_height;	//gdc input height resolution
	uint32_t input_stride;	//gdc input stride (pixel)
	uint32_t output_width;	//gdc output width resolution
	uint32_t output_height;	//gdc output height resolution
	uint32_t output_stride;	//gdc output stride (pixel)
	uint8_t div_width;	//use in dividing UV dimensions; actually a shift right
	uint8_t div_height;	//use in dividing UV dimensions; actually a shift right
	uint32_t total_planes;
	uint8_t sequential_mode;	//sequential processing
} gdc_config_t;

#ifdef __cplusplus
}
#endif

#endif //J5_VPM_INC_HB_VPM_DATA_INFO_H
