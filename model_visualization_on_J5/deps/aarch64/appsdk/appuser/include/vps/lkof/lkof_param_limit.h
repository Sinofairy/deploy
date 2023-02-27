/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2021 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/
#ifndef J5_LKOF_INCLUDE_LKOF_PARAM_LIMIT_H_
#define J5_LKOF_INCLUDE_LKOF_PARAM_LIMIT_H_
//16 BYTES FC ONE HEAD MACRO
#define FC_ONE_DST_H_MIN            (1u)
#define FC_ONE_DST_H_MAX            (1080u)

#define FC_ONE_DST_W_MIN            (1u)
#define FC_ONE_DST_W_MAX            (1920u)

#define FC_ONE_DST_TOP_MIN          (0u)
#define FC_ONE_DST_TOP_MAX          (1079u)

#define FC_ONE_DST_LEFT_MIN         (0u)
#define FC_ONE_DST_LEFT_MAX         (1919u)

#define FC_ONE_SRC_H_MIN            (1u)
#define FC_ONE_SRC_H_MAX            (1080u)

#define FC_ONE_SRC_W_MIN            (1u)
#define FC_ONE_SRC_W_MAX            (1920u)

#define FC_ONE_SRC_TOP_MIN          (0u)
#define FC_ONE_SRC_TOP_MAX          (1079u)

#define FC_ONE_SRC_LEFT_MIN         (0u)
#define FC_ONE_SRC_LEFT_MAX         (1919u)

#define FC_ONE_BLOCK_SIZE_H_MIN     (150u)
#define FC_ONE_BLOCK_SIZE_H_MAX     (200u)
#define FC_ONE_BLOCK_SIZE_H_DEFAULT     (192u)

#define FC_ONE_BLOCK_SIZE_W_MIN     (150u)
#define FC_ONE_BLOCK_SIZE_W_MAX     (200u)
#define FC_ONE_BLOCK_SIZE_W_DEFAULT     (160u)

//16 BYTES FC TWO HEAD MACRO
#define FC_TWO_OPT_EN               (1u)
#define FC_TWO_OPT_DIS              (0u)

#define FC_TWO_INITIAL_MODE_EN      (1u)
#define FC_TWO_INITIAL_MODE_DIS     (0u)

#define FC_TWO_SRC_MODE_EN          (1u)
#define FC_TWO_SRC_MODE_DIS         (0u)

#define FC_TWO_MAX_ITERATION_MIN    (1u)
#define FC_TWO_MAX_ITERATION_MAX    (10u)

#define FC_TWO_BORDER_MIN           (1u)
#define FC_TWO_BORDER_MAX           (20u)
#define FC_TWO_BORDER_DEFAULT       (8u)


#define FC_TWO_N_POINTS_MIN         (1u)
#define FC_TWO_N_POINTS_MAX         (5000u)

// end <= start
#define FC_TWO_PRY_END_MIN          (0u)
#define FC_TWO_PRY_END_MAX          (4u)

#define FC_TWO_PRY_START_MIN        (0u)
#define FC_TWO_PRY_START_MAX        (4u)

//need odd, > FC_TWO_FILTER_WIN_SIZE
#define FC_TWO_FLOW_WIN_SIZE_MIN    (3u)
#define FC_TWO_FLOW_WIN_SIZE_MAX    (23u)

#define FC_TWO_CONF_EN              (1u)
#define FC_TWO_CONF_DIS             (0u)


#define FC_TWO_FILTER_WIN_SIZE_THREE (3u)
#define FC_TWO_FILTER_WIN_SIZE_FIVE   (5u)

#define FC_TWO_FILTER_THRESHOLD_MIN   (0u)
#define FC_TWO_FILTER_THRESHOLD_MAX   (0xFFFFFFFFu)

#define FC_TWO_FILTER_K_MIN            (0u)
#define FC_TWO_FILTER_K_MAX         (0xFFFFu)

#define FC_TWO_FILTER_EN_MIN        (0u)
#define FC_TWO_FILTER_EN_MAX     (0x1Fu)

#define FC_TWO_FILTER_OUT_EN_MAX    (FC_TWO_FILTER_EN_MAX)
#define FC_TWO_FILTER_OUT_DIS       (0u)
//16 BYTES FC THREE HEAD MACRO
//16 BYTES FC FOUR HEAD MACRO
#define FC_FOUR_DST_INIT_Y_MIN      (0u)
#define FC_FOUR_DST_INIT_Y_MAX      (1079u)

#define FC_FOUR_DST_INIT_X_MIN      (0u)
#define FC_FOUR_DST_INIT_X_MAX      (1919u)

#define FC_FOUR_DST_INIT_STEP_H_MIN (1u)
#define FC_FOUR_DST_INIT_STEP_H_MAX (1079u)

#define FC_FOUR_DST_INIT_STEP_W_MIN (1u)
#define FC_FOUR_DST_INIT_STEP_W_MAX (1919u)

#define FC_FOUR_SRC_STEP_H_MIN       (1u)
#define FC_FOUR_SRC_STEP_H_MAX       (1079u)

#define FC_FOUR_SRC_STEP_W_MIN       (1u)
#define FC_FOUR_SRC_STEP_W_MAX       (1919u)

#endif // J5_LKOF_INCLUDE_LKOF_PARAM_LIMIT_H_

