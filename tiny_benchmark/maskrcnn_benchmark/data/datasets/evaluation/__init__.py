from maskrcnn_benchmark.data import datasets

from .coco import coco_evaluation
from .voc import voc_evaluation


def evaluate(dataset, predictions, output_folder, evaluate_method='',  # add by hui evaluate_method
             **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
        evaluate_method: 'coco' or 'voc' or ''(determine by dataset type)
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    # changed by hui ####################################################
    if len(evaluate_method) == 0:
        if isinstance(dataset, datasets.COCODataset):
            args.pop('voc_iou_ths')
            return coco_evaluation(**args)
        elif isinstance(dataset, datasets.PascalVOCDataset):
            return voc_evaluation(**args)
        else:
            dataset_name = dataset.__class__.__name__
            raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
    else:
        evaluate_method = evaluate_method.lower()
        if evaluate_method == 'voc':
            return voc_evaluation(**args)
        elif evaluate_method == 'coco':
            return coco_evaluation(**args)
        else:
            raise NotImplementedError("Unsupported evaluate method {}.".format(evaluate_method))
    ########################################################################################################
