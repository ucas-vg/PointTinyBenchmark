import logging

from .voc_eval import do_voc_evaluation


def voc_evaluation(dataset, predictions, output_folder, box_only, voc_iou_ths=(0.5,), **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("voc evaluation doesn't support box_only, ignored.")
    logger.info("performing voc evaluation, ignored iou_types.")
    # # changed by hui #########################################
    for th in voc_iou_ths:
        result = do_voc_evaluation(
                dataset=dataset,
                predictions=predictions,
                output_folder=output_folder,
                logger=logger,
                iou_th=th)   # add by hui
    return result
    #########################################
