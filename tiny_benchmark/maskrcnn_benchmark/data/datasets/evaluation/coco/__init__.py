from .coco_eval import do_coco_evaluation


def coco_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
    ignore_uncertain=False,
    use_iod_for_ignore=False,
    eval_standard='coco',
    gt_file=None,
    use_ignore_attr=True
):
    return do_coco_evaluation(
        dataset=dataset,
        predictions=predictions,
        box_only=box_only,
        output_folder=output_folder,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        ignore_uncertain=ignore_uncertain,
        use_iod_for_ignore=use_iod_for_ignore,
        eval_standard=eval_standard,
        gt_file=gt_file,
        use_ignore_attr=use_ignore_attr
    )
