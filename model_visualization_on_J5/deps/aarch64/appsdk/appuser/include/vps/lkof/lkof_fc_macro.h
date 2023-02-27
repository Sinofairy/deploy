/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2021 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/
#ifndef LKOF_FC_MACRO_H
#define LKOF_FC_MACRO_H
//16 BYTES FC ONE HEAD MACRO
#define FC_ONE_32_DST_H            (0x1FFFu)
#define FC_ONE_32_DST_H_SB         (0)
#define FC_ONE_32_DST_W            (0x1FFFu)
#define FC_ONE_32_DST_W_SB         (13)
#define FC_ONE_32_DST_TOP          (0x3Fu)
#define FC_ONE_32_DST_TOP_SB       (26)

#define FC_ONE_64_DST_TOP          (0x1FC0u)
#define FC_ONE_64_DST_TOP_SB       (6)
#define FC_ONE_64_DST_LEFT         (0x1FFFu)
#define FC_ONE_64_DST_LEFT_SB      (7)
#define FC_ONE_64_SRC_H            (0xFFFu)
#define FC_ONE_64_SRC_H_SB         (20)

#define FC_ONE_96_SRC_H            (0x1000u)
#define FC_ONE_96_SRC_H_SB         (12)
#define FC_ONE_96_SRC_W            (0x1FFFu)
#define FC_ONE_96_SRC_W_SB         (1)
#define FC_ONE_96_SRC_TOP          (0x1FFFu)
#define FC_ONE_96_SRC_TOP_SB       (14)
#define FC_ONE_96_SRC_LEFT         (0x1Fu)
#define FC_ONE_96_SRC_LEFT_SB      (27)

#define FC_ONE_128_SRC_LEFT        (0x1FE0u)
#define FC_ONE_128_SRC_LEFT_SB     (5)
#define FC_ONE_128_BLOCK_SIZE_H    (0xFFFu)
#define FC_ONE_128_BLOCK_SIZE_H_SB (8)
#define FC_ONE_128_BLOCK_SIZE_W    (0xFFFu)
#define FC_ONE_128_BLOCK_SIZE_W_SB (20)

//16 BYTES FC TWO HEAD MACRO
#define FC_TWO_32_OPT_EN            (0x1u)
#define FC_TWO_32_OPT_EN_SB         (8)

#define FC_TWO_32_INITIAL_MODE            (0x1u)
#define FC_TWO_32_INITIAL_MODE_SB         (9)

#define FC_TWO_32_SRC_MODE            (0x1u)
#define FC_TWO_32_SRC_MODE_SB         (10)

#define FC_TWO_32_MAX_ITERATION            (0xFu)
#define FC_TWO_32_MAX_ITERATION_SB         (11)

#define FC_TWO_32_BORDER            (0xFFu)
#define FC_TWO_32_BORDER_SB         (15)

#define FC_TWO_32_N_POINTS            (0x1FFu)
#define FC_TWO_32_N_POINTS_SB         (23)

#define FC_TWO_64_N_POINTS            (0xFE00u)
#define FC_TWO_64_N_POINTS_SB         (9)
#define FC_TWO_64_PRY_END            (0x7u)
#define FC_TWO_64_PRY_END_SB         (7)
#define FC_TWO_64_PRY_START            (0x7u)
#define FC_TWO_64_PRY_START_SB         (10)
#define FC_TWO_64_FLOW_WIN_SIZE            (0x1Fu)
#define FC_TWO_64_FLOW_WIN_SIZE_SB         (13)
#define FC_TWO_64_CONF_EN            (0x1u)
#define FC_TWO_64_CONF_EN_SB         (18)
#define FC_TWO_64_FILTER_OUT_EN            (0x1Fu)
#define FC_TWO_64_FILTER_OUT_EN_SB         (19)
#define FC_TWO_64_FILTER_WIN_SIZE            (0x7u)
#define FC_TWO_64_FILTER_WIN_SIZE_SB         (24)
#define FC_TWO_64_FILTER_THRESHOLD            (0x1Fu)
#define FC_TWO_64_FILTER_THRESHOLD_SB         (27)

#define FC_TWO_96_FILTER_THRESHOLD            (0xFFFFFFE0u)
#define FC_TWO_96_FILTER_THRESHOLD_SB         (5)

#define FC_TWO_96_FILTER_K            (0x1Fu)
#define FC_TWO_96_FILTER_K_SB         (27)

#define FC_TWO_128_FILTER_K            (0xFFFFFFE0u)
#define FC_TWO_128_FILTER_K_SB         (5)
#define FC_TWO_128_FILTER_EN            (0x1Fu)
#define FC_TWO_128_FILTER_EN_SB         (27)
//16 BYTES FC THREE HEAD MACRO
//16 BYTES FC FOUR HEAD MACRO
#define FC_FOUR_32_DST_INIT_Y            (0xFFFFFu)
#define FC_FOUR_32_DST_INIT_Y_SB         (8)
#define FC_FOUR_32_DST_INIT_X            (0xFu)
#define FC_FOUR_32_DST_INIT_X_SB         (28)

#define FC_FOUR_64_DST_INIT_X            (0xFFFFu)
#define FC_FOUR_64_DST_INIT_X_SB         (4)
#define FC_FOUR_64_DST_INIT_STEP_H       (0xFFFFu)
#define FC_FOUR_64_DST_INIT_STEP_H_SB         (16)

#define FC_FOUR_96_DST_INIT_STEP_H           (0xF0000u)
#define FC_FOUR_96_DST_INIT_STEP_H_SB         (16)
#define FC_FOUR_96_DST_INIT_STEP_W           (0xFFFFFu)
#define FC_FOUR_96_DST_INIT_STEP_W_SB         (4)
#define FC_FOUR_96_SRC_STEP_H            (0xFFu)
#define FC_FOUR_96_SRC_STEP_H_SB         (24)

#define FC_FOUR_128_SRC_STEP_H           (0xFFFF00u)
#define FC_FOUR_128_SRC_STEP_H_SB        (8)
#define FC_FOUR_128_SRC_STEP_W           (0xFFFFFu)
#define FC_FOUR_128_SRC_STEP_W_SB         (12)

//POINTS INFORMATION
#define FIRST_32_POINTS_X                  (0xFFFFFu)
#define FIRST_32_POINTS_X_SB                  (0)
#define FIRST_32_POINTS_Y                  (0xFFFu)
#define FIRST_32_POINTS_Y_SB                  (20)

#define FIRST_64_POINTS_Y                  (0xFF000u)
#define FIRST_64_POINTS_Y_SB                  (12)
#define FIRST_64_POINTS_STATUS                  (0x3u)
#define FIRST_64_POINTS_STATUS_SB                  (8)

#define SECOND_64_POINTS_X                  (0xFFFFFu)
#define SECOND_64_POINTS_X_SB                  (10)
#define SECOND_64_POINTS_Y                  (0x3u)
#define SECOND_64_POINTS_Y_SB                  (30)

#define SECOND_96_POINTS_Y                  (0xFFFFC)
#define SECOND_96_POINTS_Y_SB                  (2)
#define SECOND_96_POINTS_STATUS                  (0x3u)
#define SECOND_96_POINTS_STATUS_SB                  (18)

#define THIRD_96_POINTS_X                  (0xFFFu)
#define THIRD_96_POINTS_X_SB                  (20)

#define THIRD_128_POINTS_X                  (0xFF000u)
#define THIRD_128_POINTS_X_SB                  (12)
#define THIRD_128_POINTS_Y                  (0xFFFFFu)
#define THIRD_128_POINTS_Y_SB                  (8)
#define THIRD_128_POINTS_STATUS                  (0x3u)
#define THIRD_128_POINTS_STATUS_SB                  (28)
#define THIRD_128_POINTS_STATUS_RESERVED                  (0x3u)
#define THIRD_128_POINTS_STATUS_RESERVED_SB                  (30)
#endif //LKOF_FC_MACRO_H

