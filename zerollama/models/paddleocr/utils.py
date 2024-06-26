from easydict import EasyDict as edict

engine_args = edict({
    "use_gpu": True,
    "use_xpu": False,
    "use_npu": False,
    "use_mlu": False,
    "ir_optim": False,
    "use_tensorrt": False,
    "min_subgraph_size": 15,
    "precision": "fp32",
    "gpu_mem": 500,
    "gpu_id": 0
})
text_detector_args = edict({
    "det_model_dir": "",
    "det_limit_side_len": 960,
    "det_limit_type": "max",
    "det_box_type": "quad"
})
multi_process_args = edict({
    "use_mp": False,
    "total_process_num": 1,
    "process_id": 0,
    "benchmark": False,
    "save_log_path": "",
    "show_log": False,
    "use_onnx": False
})
text_recognizer_args = edict({
    "rec_model_dir": "",
    "rec_image_inverse": True,
    "rec_image_shape": "3, 48, 320",
    "rec_batch_num": 6,
    "max_text_length": 25,
    "rec_char_dict_path": None,
    "use_space_char": True,
    "vis_font_path": "./doc/fonts/simfang.ttf",
    "drop_score": 0.5,
})
layout_args = edict({
    "layout_model_dir": "",
    "layout_dict_path": None,
    "layout_score_threshold": 0.5,
    "layout_nms_threshold": 0.5,
})
tabel_args = edict({
    "table_model_dir": "",
    "table_char_dict_path": None,
    "table_max_len": 488,
    "table_algorithm": "TableAttn",
    "merge_no_span_structure": True,
})
