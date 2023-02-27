import os

from easydict import EasyDict


root = "/pilot_data"

# ----- Required -----

datapaths = dict(
    rear_plate_detection=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/rear_plate_detection/train.rec",  # noqa
                anno_path=f"{root}/rear_plate_detection/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    person_face_detection=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/person_face_detection/train.rec",  # noqa
                anno_path=f"{root}/person_face_detection/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    person_occlusion_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/person_occlusion_classification/train.rec",  # noqa
                anno_path=f"{root}/person_occlusion_classification/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    vehicle_occlusion_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/vehicle_occlusion_classification/train.rec",  # noqa
                anno_path=f"{root}/vehicle_occlusion_classification/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    rear_occlusion_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/rear_occlusion_classification/train.rec",  # noqa
                anno_path=f"{root}/rear_occlusion_classification/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    rear_part_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/rear_part_classification/train.rec",  # noqa
                anno_path=f"{root}/rear_part_classification/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    person_pose_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/person_pose_classification/train.rec",  # noqa
                anno_path=f"{root}/person_pose_classification/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    vehicle_3d_detection=dict(
        train_batch_size_per_ctx=12,
        train_data_paths=[
            dict(
                # hard case
                rec_path=[
                    f"{root}/vehicle_3d_detection/train.rec",
                ],
                anno_path=[
                    f"{root}/vehicle_3d_detection/train.anno.pb_rec",  # noqa
                ],
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=10,
        val_data_paths=None,
    ),
    person_3d_detection=dict(
        train_batch_size_per_ctx=16,
        train_data_paths=[
            dict(
                rec_path=[
                    f"{root}/person_3d_detection/train.rec",  # noqa
                ],
                anno_path=[
                    f"{root}/person_3d_detection/train.anno.pb_rec",  # noqa
                ],
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=10,
        val_data_paths=None,
    ),
    vehicle_category=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/vehicle_category_classification/train.rec",  # noqa
                anno_path=f"{root}/vehicle_category_classification/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=16,
        val_data_paths=None,
    ),
    vehicle_wheel_kps=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/vehicle_wheel_kps/train.rec",  # noqa
                anno_path=f"{root}/vehicle_wheel_kps/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=16,
        val_data_paths=None,
    ),
    vehicle_ground_line=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/vehicle_ground_line/train.rec",  # noqa
                anno_path=f"{root}/vehicle_ground_line/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=16,
        val_data_paths=None,
    ),
    vehicle_wheel_detection=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/vehicle_wheel_detection/train.rec",  # noqa
                anno_path=f"{root}/vehicle_wheel_detection/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=16,
        val_data_paths=None,
    ),
    person_orientation_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/person_orientation_classification/train.rec",
                anno_path=f"{root}/person_orientation_classification/train.anno.pb_rec", 
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    cyclist=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/cyclist_detection/train.rec",
                anno_path=f"{root}/cyclist_detection/train.anno.pb_rec",
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    person=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/person_detection/train.rec",
                anno_path=f"{root}/person_detection/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    vehicle=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/vehicle_detection/train.rec",  # noqa
                anno_path=f"{root}/vehicle_detection/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    rear=dict(
        train_batch_size_per_ctx=24,
        train_data_paths=[
            dict(
                rec_path=f"{root}/vehicle_rear_detection/train.rec",
                anno_path=f"{root}/vehicle_rear_detection/train.anno.pb_rec",
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    vehicle_truncation_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{root}/vehicle_truncation_classification/train.rec",  # noqa
                anno_path=f"{root}/vehicle_truncation_classification/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=16,
        val_data_paths=None,
    ),
    semantic_parsing=dict(
        train_batch_size_per_ctx=16,
        train_data_paths=[
            dict(
                rec_path=f"{root}/default_parsing/train.rec",  # noqa
                anno_path=f"{root}/default_parsing/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
    lane_parsing=dict(
        train_batch_size_per_ctx=16,
        train_data_paths=[
            dict(
                rec_path=f"{root}/lane_parsing/train.rec",  # noqa
                anno_path=f"{root}/lane_parsing/train.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
)

datapaths = EasyDict(datapaths)
