import os

from easydict import EasyDict

root = "/nas/lmd/workspace/data/data/tmp_out"
# ----- Required -----

datapaths = dict(
    vehicle_detection=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/vehicle_detection/idx",  # noqa
                    img_path=f"{root}/vehicle_detection/img",  # noqa
                    anno_path=f"{root}/vehicle_detection/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    cyclist_detection=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/cyclist_detection/idx",  # noqa
                    img_path=f"{root}/cyclist_detection/img",  # noqa
                    anno_path=f"{root}/cyclist_detection/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    person_detection=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/person_detection/idx",  # noqa
                    img_path=f"{root}/person_detection/img",  # noqa
                    anno_path=f"{root}/person_detection/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    default_parsing=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/default_parsing/idx",  # noqa
                    img_path=f"{root}/default_parsing/img",  # noqa
                    anno_path=f"{root}/default_parsing/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    lane_parsing=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/lane_parsing/idx",  # noqa
                    img_path=f"{root}/lane_parsing/img",  # noqa
                    anno_path=f"{root}/lane_parsing/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    person_face_detection=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/person_face_detection/idx",  # noqa
                    img_path=f"{root}/person_face_detection/img",  # noqa
                    anno_path=f"{root}/person_face_detection/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    person_occlusion_classification=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/person_occlusion_classification/idx",  # noqa
                    img_path=f"{root}/person_occlusion_classification/img",  # noqa
                    anno_path=f"{root}/person_occlusion_classification/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    person_orientation_classification=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/person_orientation_classification/idx",  # noqa
                    img_path=f"{root}/person_orientation_classification/img",  # noqa
                    anno_path=f"{root}/person_orientation_classification/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    person_pose_classification=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/person_pose_classification/idx",  # noqa
                    img_path=f"{root}/person_pose_classification/img",  # noqa
                    anno_path=f"{root}/person_pose_classification/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    rear_detection=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/rear_detection/idx",  # noqa
                    img_path=f"{root}/rear_detection/img",  # noqa
                    anno_path=f"{root}/rear_detection/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    rear_occlusion_classification=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/rear_occlusion_classification/idx",  # noqa
                    img_path=f"{root}/rear_occlusion_classification/img",  # noqa
                    anno_path=f"{root}/rear_occlusion_classification/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    rear_part_classification=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/rear_part_classification/idx",  # noqa
                    img_path=f"{root}/rear_part_classification/img",  # noqa
                    anno_path=f"{root}/rear_part_classification/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    rear_plate_detection=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/rear_plate_detection/idx",  # noqa
                    img_path=f"{root}/rear_plate_detection/img",  # noqa
                    anno_path=f"{root}/rear_plate_detection/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    vehicle_category_classification=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/vehicle_category_classification/idx",  # noqa
                    img_path=f"{root}/vehicle_category_classification/img",  # noqa
                    anno_path=f"{root}/vehicle_category_classification/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    vehicle_ground_line=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/vehicle_ground_line/idx",  # noqa
                    img_path=f"{root}/vehicle_ground_line/img",  # noqa
                    anno_path=f"{root}/vehicle_ground_line/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    vehicle_occlusion_classification=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/vehicle_occlusion_classification/idx",  # noqa
                    img_path=f"{root}/vehicle_occlusion_classification/img",  # noqa
                    anno_path=f"{root}/vehicle_occlusion_classification/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    vehicle_truncation_classification=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/vehicle_truncation_classification/idx",  # noqa
                    img_path=f"{root}/vehicle_truncation_classification/img",  # noqa
                    anno_path=f"{root}/vehicle_truncation_classification/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    vehicle_wheel_detection=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/vehicle_wheel_detection/idx",  # noqa
                    img_path=f"{root}/vehicle_wheel_detection/img",  # noqa
                    anno_path=f"{root}/vehicle_wheel_detection/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
    vehicle_wheel_kps=dict(
        dict(
            train_data_paths=[
                dict(
                    idx_path=f"{root}/vehicle_wheel_kps/idx",  # noqa
                    img_path=f"{root}/vehicle_wheel_kps/img",  # noqa
                    anno_path=f"{root}/vehicle_wheel_kps/anno",  # noqa
                    sample_weight=16,
                )
            ],
        ),
    ),
)

datapaths = EasyDict(datapaths)
buckets = [
    "matrix",
]
