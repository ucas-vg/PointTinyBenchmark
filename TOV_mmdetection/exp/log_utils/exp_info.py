_log_root= "mmdetection_cache/work_dir/locnet/"

noisept_pathes = {
    "TinyPerson": {
        "faster": {  # faster as estimator, locator
            "uniform_pseuwh16_round0": _log_root + "TinyPerson/noisept1/faster_rcnn_r50_fpn_1x_TinyPerson640x512Noisept/pseuwh16_old640x512_lr004_1x_8g",
            "uniform_pseuwh16_round1": _log_root + "TinyPerson/noisept1/faster_rcnn_r50_fpn_1x_TinyPerson640x512Noisept/pseuwh16_round1_old640x512_lr004_1x_8g",
            "uniform_pseuwh16_round2": _log_root + "TinyPerson/noisept1/faster_rcnn_r50_fpn_1x_TinyPerson640x512Noisept/pseuwh16_round2_old640x512_lr004_1x_8g",
            "uniform_pseuwh16_round3": _log_root + "TinyPerson/noisept1/faster_rcnn_r50_fpn_1x_TinyPerson640x512Noisept/pseuwh16_round3_old640x512_lr004_1x_8g",

            "uniform_pseuwh32_round0": _log_root + "TinyPerson/noisept1/faster_rcnn_r50_fpn_1x_TinyPerson640x512Noisept/pseuwh32_old640x512_lr004_1x_8g",
            "uniform_pseuwh32_round1": _log_root + "TinyPerson/noisept1/faster_rcnn_r50_fpn_1x_TinyPerson640x512Noisept/pseuwh32_round1_old640x512_lr004_1x_8g",
            "uniform_pseuwh32_round2": _log_root + "TinyPerson/noisept1/faster_rcnn_r50_fpn_1x_TinyPerson640x512Noisept/pseuwh32_round2_old640x512_lr004_1x_8g",

            "uniform_pseuwh64_round0": _log_root + "TinyPerson/noisept1/faster_rcnn_r50_fpn_1x_TinyPerson640x512Noisept/pseuwh64_old640x512_lr004_1x_8g",
            "uniform_pseuwh64_round1": _log_root + "TinyPerson/noisept1/faster_rcnn_r50_fpn_1x_TinyPerson640x512Noisept/pseuwh64_round1_old640x512_lr004_1x_8g",
        },

        "reppoint": {  # reppoint as estimator, locator
            "uniform_pseuwh16_round0": _log_root + "TinyPerson/noisept1/reppoints_moment_r50_fpns4_gn-neck+head_1x_TinyPerson640_bboxE_scale4/pseuwh32_old640x512_lr004_1x_8g",
            "uniform_pseuwh16_round1": _log_root + "TinyPerson/noisept1/reppoints_moment_r50_fpns4_gn-neck+head_1x_TinyPerson640_bboxE_scale4/pseuwh32_round1_old640x512_lr004_1x_8g",
            "uniform_pseuwh16_round2": _log_root + "TinyPerson/noisept1/reppoints_moment_r50_fpns4_gn-neck+head_1x_TinyPerson640_bboxE_scale4/pseuwh32_round2_old640x512_lr004_1x_8g",

            "uniform_pseuwh32_round0": _log_root + "TinyPerson/noisept1/reppoints_moment_r50_fpns4_gn-neck+head_1x_TinyPerson640_bboxE_scale4/pseuwh32_old640x512_lr004_1x_8g",
        },

        "reppoint1_faster": {  #
            "uniform_pseuwh16_round1": _log_root + "TinyPerson/noisept1/reppoint1_faster/faster_rcnn_r50_fpn_1x_TinyPerson640x512Noisept/pseuwh16_round1_old640x512_lr004_1x_8g",
        }
    },
    "VisDronePerson": {
        "reppoint": {
            "uniform_pseuwh32_round0": _log_root + "noisept1/reppoints_moment_r50_fpn_gn-neck+head_1x_visDroneNoisept640_bboxE/VisDrone2018-DET-train-person_noisept/pseuwh32_640_lr0.01_8e11e12e",
            "uniform_pseuwh32_round1": _log_root + "noisept1/reppoints_moment_r50_fpn_gn-neck+head_1x_visDroneNoisept640_bboxE/VisDrone2018-DET-train-person_noisept/pseuwh32_round1_640_lr0.01_8e11e12e",
            "uniform_pseuwh32_round2": _log_root + "noisept1/reppoints_moment_r50_fpn_gn-neck+head_1x_visDroneNoisept640_bboxE/VisDrone2018-DET-train-person_noisept/pseuwh32_round2_640_lr0.01_8e11e12e",
            "uniform_pseuwh32_round3": _log_root + "noisept1/reppoints_moment_r50_fpn_gn-neck+head_1x_visDroneNoisept640_bboxE/VisDrone2018-DET-train-person_noisept/pseuwh32_round3_640_lr0.01_8e11e12e",

            "rg0.26_0.125_pseuwh32_round0": _log_root + "noisept1/reppoints_moment_r50_fpn_gn-neck+head_1x_visDroneNoisept640_bboxE/VisDrone2018-DET-train-person_rg-0.26_0.125_noisept/pseuwh32_640_lr0.01_8e11e12e",
            "rg0.26_0.125_pseuwh32_round1": _log_root + "noisept1/reppoints_moment_r50_fpn_gn-neck+head_1x_visDroneNoisept640_bboxE/VisDrone2018-DET-train-person_rg-0.26_0.125_noisept/pseuwh32_round1_640_lr0.01_8e11e12e",
            "rg0.26_0.125_pseuwh32_round2": _log_root + "noisept1/reppoints_moment_r50_fpn_gn-neck+head_1x_visDroneNoisept640_bboxE/VisDrone2018-DET-train-person_rg-0.26_0.125_noisept/pseuwh32_round2_640_lr0.01_8e11e12e",

            "rg0.28_0.167_pseuwh32_round0": _log_root + "noisept1/reppoints_moment_r50_fpn_gn-neck+head_1x_visDroneNoisept640_bboxE/VisDrone2018-DET-train-person_rg-0.28_0.167_noisept/pseuwh32_640_lr0.01_8e11e12e",
            "rg0.28_0.167_pseuwh32_round1": _log_root + "noisept1/reppoints_moment_r50_fpn_gn-neck+head_1x_visDroneNoisept640_bboxE/VisDrone2018-DET-train-person_rg-0.28_0.167_noisept/pseuwh32_round1_640_lr0.01_8e11e12e",
            "rg0.28_0.167_pseuwh32_round2": _log_root + "noisept1/reppoints_moment_r50_fpn_gn-neck+head_1x_visDroneNoisept640_bboxE/VisDrone2018-DET-train-person_rg-0.28_0.167_noisept/pseuwh32_round2_640_lr0.01_8e11e12e",

        }
    },
}

gen_anns = {
    "TinyPerson": {
        "faster": {
            "uniform_pseuwh16_round1": "tiny_set/mini_annotations/noise/round_faster/tiny_set_train_sw640_sh512_all_erase_noisept_pseuw16h16_round1.json",
        }
    }
}
