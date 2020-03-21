# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from math import inf


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    test_func,      # add by hui
    cfg,            # add by hui
    distributed     # add by hui
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
        # ################################################## add by hui ###############################################
        last_test_iter = inf if cfg.SOLVER.TEST_ITER_RANGE[1] < 0 else cfg.SOLVER.TEST_ITER_RANGE[1]
        if cfg.SOLVER.TEST_ITER > 0 and (iteration + 1) % cfg.SOLVER.TEST_ITER == 0\
                and cfg.SOLVER.TEST_ITER_RANGE[0] <= (iteration + 1) <= last_test_iter:
            test_func(cfg, model, distributed)
            model.train()
            evaluate_more(cfg)
        ###############################################################################################################

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    if cfg.TEST_FINAL_ITER:
        test_func(cfg, model, distributed)
        evaluate_more(cfg)

# ################################################## add by hui ###############################################


def evaluate_more(cfg):
    print(cfg.DATASETS.TEST)
    for dataset_name in cfg.DATASETS.TEST:
        from third.Cityscapes.cityperson_eval import cityperson_eval
        from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
        print("dataset: ", DatasetCatalog.DATASETS[dataset_name])

        det_file_path = cfg.OUTPUT_DIR + '/inference/{}/bbox.json'.format(dataset_name)
        gt_file_path = DatasetCatalog.DATA_DIR + '/' + DatasetCatalog.DATASETS[dataset_name]["ann_file"]
        if 'tiny' in dataset_name:
            CUT_WH = None
            if cfg.TEST.MERGE_RESULTS:
                # _, det_file_path = COCOMergeResult(use_nms=True, nms_th=0.5)(
                #     gt_file_path,
                #     det_file_path,
                #     cfg.OUTPUT_DIR + '/inference/{}/'.format(dataset_name)
                # )
                det_file_path = cfg.OUTPUT_DIR + '/inference/{}/bbox_merge_nms0.5.json'.format(dataset_name)
                gt_file_path = cfg.TEST.MERGE_GT_FILE
                CUT_WH = (1, 1)
            else:
                # get corner dataset cut piece
                if dataset_name.find('corner') != -1:
                    idx = dataset_name.find('corner_') + len('corner_')
                    sub_strs = dataset_name[idx:].split('_')
                    print(sub_strs)
                    pw, ph = int(sub_strs[0][2:]), int(sub_strs[1][2:])
                    # assert sub_strs[0][:2] == 'pw' and sub_strs[1][:2] == 'ph', \
                    #     'only support cut by piece format corner dataset.'
                    if sub_strs[0][:2] == 'pw' and sub_strs[1][:2] == 'ph':
                        CUT_WH = (pw, ph)
                    elif sub_strs[0][:2] == 'sw' and sub_strs[1][:2] == 'sh':
                        CUT_WH = (1, 1)
                        assert cfg.TEST.MERGE_RESULTS
                    else:
                        assert False
            cityperson_eval(det_file_path, gt_file_path, CUT_WH=CUT_WH, ignore_uncertain=cfg.TEST.IGNORE_UNCERTAIN,
                            use_iod_for_ignore=cfg.TEST.USE_IOD_FOR_IGNORE, use_citypersons_standard=False)
        elif 'pedestrian' in dataset_name:
            cityperson_eval(det_file_path, gt_file_path)

###############################################################################################################
