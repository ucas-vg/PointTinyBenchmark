# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import shutil

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine import inference_trainer
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog


def fixed_seed(init_seed):  # add by hui
    import numpy as np
    import random
    if init_seed == -2:
        init_seed = np.random.randint(0, 2**32)
    print("set random seed to {}.".format(init_seed))
    random.seed(init_seed)
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)

    # torch.backends.cudnn.enabled = False
    # # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def train(cfg, local_rank, distributed):
    # ############################# add by hui ##########################
    if cfg.FIXED_SEED >= 0 or cfg.FIXED_SEED == -2:
        fixed_seed(cfg.FIXED_SEED)
    # ###################################################################

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    # ############################## add by hui #######################
    print(cfg.MODEL.WEIGHT)
    print(checkpointer.has_checkpoint())
    # pretrain_checkpoint = torch.load(cfg.MODEL.WEIGHT, map_location=torch.device("cpu"))
    ##################################################################
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    # ################################################ change by hui ################################################
    inference_trainer.do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        test_func=run_test,
        cfg=cfg,
        distributed=distributed
    )
    ################################################################################################

    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON or cfg.MODEL.GAU_ON else cfg.MODEL.RPN_ONLY,  # changed for fcos
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            ignore_uncertain=cfg.TEST.IGNORE_UNCERTAIN,
            use_iod_for_ignore=cfg.TEST.USE_IOD_FOR_IGNORE,
            eval_standard=cfg.TEST.COCO_EVALUATE_STANDARD,
            use_last_prediction=cfg.TEST.DEBUG.USE_LAST_PREDICTION,
            evaluate_method=cfg.TEST.EVALUATE_METHOD,
            voc_iou_ths=cfg.TEST.VOC_IOU_THS,
            gt_file={'merge': cfg.TEST.MERGE_GT_FILE,
                     'sub': DatasetCatalog.DATA_DIR + '/' + DatasetCatalog.DATASETS[dataset_name]["ann_file"]},
            use_ignore_attr=cfg.TEST.USE_IGNORE_ATTR
        )
        synchronize()

# ################################################ add by hui #################################################


def adaptive_config_change(name, old, new):
    if old == new:
        return
    print('    {:<20} {} --> {}'.format(name, old, new))
    cfg.merge_from_list([name, new])

# #################################################################################################

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
        default=True  # add by hui
    )
    # ################################################ add by hui #################################################
    parser.add_argument(
        "--temp",
        help="whether generate to temp output",
        default=False,
        type=bool
    )
    # #################################################################################################
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # ################### change by hui #################################################
    if args.temp:
        if os.path.exists("./outputs/temp"): shutil.rmtree('./outputs/temp')
        adaptive_config_change("OUTPUT_DIR", cfg.OUTPUT_DIR, './outputs/temp')
    cfg.freeze()

    some_pre_deal()
    ##################################################################################################

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)

# ################################################ add by hui #################################################


def some_pre_deal():
    """add by hui"""
    from PIL import ImageFile    # add by hui
    ImageFile.LOAD_TRUNCATED_IMAGES = True   # add by hui
    assert cfg.SOLVER.NUM_GPU == torch.cuda.device_count(), 'NUM_GPU is not equal to visible GPU count {} vs {}.'\
        .format(cfg.SOLVER.NUM_GPU, torch.cuda.device_count())
##################################################################################################


if __name__ == "__main__":
    main()
