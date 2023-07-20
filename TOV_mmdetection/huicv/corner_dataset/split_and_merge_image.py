import numpy as np
from copy import deepcopy
import json
import os
import cv2
from PIL import Image
from tqdm import tqdm


class SplitImage(object):

    def __init__(self, use_fix_size=False, use_least_pieces_size=False):
        self.use_fix_size = use_fix_size
        self.use_least_pieces_size = use_least_pieces_size

    def __get_corners(self, image_size, piece, stride, overlap):
        def left_move(image_size, corners):
            iw, ih = image_size
            new_corners = []
            for sx, sy, ex, ey in corners.reshape(-1, 4):
                if ex > iw:
                    sx -= (ex - iw)
                    ex = iw
                if ey > ih:
                    sy -= (ey - ih)
                    ey = ih
                sx, sy = max(0, sx), max(0, sy)
                new_corners.append([sx, sy, ex, ey])
            return np.array(new_corners).reshape(corners.shape)

        corners = []
        for x in range(piece[0]):
            col_corners = []
            for y in range(piece[1]):
                sx, sy = np.array([x, y]) * stride
                ex, ey = (np.array([x, y]) + 1) * stride + overlap
                col_corners.append([sx, sy, ex, ey])
            corners.append(col_corners)
        corners = np.array(corners)
        if self.use_fix_size:
            corners = left_move(image_size, corners)
        else:
            corners[:, :, [0, 2]] = np.clip(corners[:, :, [0, 2]], 0, image_size[0])
            corners[:, :, [1, 3]] = np.clip(corners[:, :, [1, 3]], 0, image_size[1])
        return corners.transpose((1, 0, 2))

    def __get_sub_image_corners_by_pieces(self, image_size, pieces=(1, 1), overlap=(0, 0)):
        """
        :param image_size: tuple, (w, h) of origin image .
        :param pieces: tuple, (pw, ph), how many pieces to split image.
        :param overlap: tuple, (ow, oh), overlap of two sub neighbour image.
        :return: np.ndarray, (x1, y1, x2, y2), shape=(ph, pw, 4), corner of sub images.
        """
        size, pieces, overlap = np.array(image_size), np.array(list(pieces)), np.array(list(overlap))
        stride = np.ceil((size - overlap) / pieces).astype(np.int)
        return self.__get_corners(image_size, pieces, stride, overlap)

    def __get_sub_image_corners_by_size(self, image_size, sub_image_size, overlap=(0, 0)):
        """
        :param image_size: tuple, (w, h) of origin image .
        :param sub_image_size: tuple, (sw, sh), size of sub image.
        :param overlap: tuple, (ow, oh), overlap of two sub neighbour image.
        :return: np.ndarray, (x1, y1, x2, y2), shape=(ph, pw, 4), corner of sub images.
        """

        def least_pieces_size(image_size, size):
            c1 = np.ceil(image_size[0] / size[0]) * np.ceil(image_size[1] / size[1])
            c2 = np.ceil(image_size[0] / size[1]) * np.ceil(image_size[1] / size[0])
            if c1 <= c2:
                return size
            else:
                return size[1], size[0]

        if self.use_least_pieces_size:
            sub_image_size = least_pieces_size(image_size, sub_image_size)

        size, sub_image_size, overlap = np.array(image_size), np.array(list(sub_image_size)), np.array(list(overlap))
        stride = np.ceil(sub_image_size - overlap).astype(np.int)
        piece = np.ceil((size - overlap) / (sub_image_size - overlap)).astype(int)
        return self.__get_corners(image_size, piece, stride, overlap)

    def get_sub_image_corners(self, image_size, pieces=None, sub_image_size=None, overlap=(0, 0)):
        """
            cut function for image
        :param image_size: tuple, (w, h) of origin image .
        :param pieces: tuple, (pw, ph), how many pieces to split image.
        :param sub_image_size: tuple, (sw, sh), size of sub image.
        :param overlap: tuple, (ow, oh), overlap of two sub neighbour image.
        :return: np.ndarray, (x1, y1, x2, y2), shape=(ph, pw, 4), corner of sub images.
        """
        assert (pieces is None) ^ (sub_image_size is None), \
            '"pieces" and "sub_image_size" can only specified one of them. but got {} and {}'.format(
                pieces, sub_image_size)
        if sub_image_size is not None:
            return self.__get_sub_image_corners_by_size(image_size, sub_image_size, overlap)
        elif pieces is not None:
            return self.__get_sub_image_corners_by_pieces(image_size, pieces, overlap)

    def get_sub_image_boxes(self, corner: np.ndarray, annos: np.ndarray, annos_area: np.ndarray, keep_overlap=0.7):
        """
            cut function for annotation
        :param corner: sub image's corner
        :param annos: annotations in origin image
        :param annos_area: annotations's area in origin image
        :param keep_overlap: IOU(annotation in sub image, annotation in origin image) > keep_overlap will keep, or ignore.
        :return:
            annos: annos in sub_image
            annos_id[keep]: correspounding annos id in origin image
        """
        annos = annos.copy()
        annos_id = np.array(range(len(annos)))
        annos[:, [0, 2]] = np.clip(annos[:, [0, 2]], corner[0], corner[2])
        annos[:, [1, 3]] = np.clip(annos[:, [1, 3]], corner[1], corner[3])
        new_anno_area = np.sqrt((annos[:, 2] - annos[:, 0]) * (annos[:, 3] - annos[:, 1]))
        keep = (new_anno_area / annos_area) >= keep_overlap
        annos = annos[keep]
        annos = (annos.reshape((-1, 2)) - corner[:2]).reshape((-1, 4))
        return annos, annos_id[keep]

    def cut_image(self, image: np.ndarray, annos=None, pieces=None, sub_image_size=None, overlap=(0, 0),
                  anno_keep_overlap=0.7):
        h, w, c = image.shape
        sub_images = []
        corners = self.get_sub_image_corners((w, h), pieces, sub_image_size, overlap)
        corners = corners.reshape((-1, 4))
        for (x1, y1, x2, y2) in corners:
            sub_images.append(image[y1:y2, x1:x2].copy())

        if annos is not None:
            annos_area = np.sqrt((annos[:, 2] - annos[:, 0]) * (annos[:, 3] - annos[:, 1]))
            sub_annos, sub_keeps = [], []
            for corner in corners:
                sub_anno, keep = self.get_sub_image_boxes(corner, annos, annos_area, anno_keep_overlap)
                sub_annos.append(sub_anno)
                sub_keeps.append(keep)
            return sub_images, corners, sub_annos, sub_keeps
        return sub_images, corners, None, None

    def get_sub_image_corner_and_boxes(self, image_size, bboxes: np.ndarray, pieces=None,
                                       sub_image_size=None, overlap=(0, 0), anno_keep_overlap=0.7):
        corners = self.get_sub_image_corners(image_size, pieces, sub_image_size, overlap)
        corners = corners.reshape((-1, 4))
        if len(bboxes) > 0:
            annos_area = np.sqrt((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]))
            sub_annos, sub_keeps = [], []
            for corner in corners:
                sub_anno, keep = self.get_sub_image_boxes(corner, bboxes, annos_area, anno_keep_overlap)
                sub_annos.append(sub_anno)
                sub_keeps.append(keep)
        else:
            sub_annos = [np.array([]).reshape((0, 4)) for _ in corners]
            sub_keeps = [np.array([]).reshape((0,)) for _ in corners]
        return corners, sub_annos, sub_keeps


class MergeResult(object):
    def __init__(self, use_nms=True, nms_th=0.5):
        self.use_nms = use_nms
        self.nms_th = nms_th

    def merge_result(self, corners, results, scores=None):
        merge_result = self.translate_bboxes(corners, results)
        no_empty_result = sum([len(result) > 0 for result in results])
        if no_empty_result > 1 and self.use_nms:  # only when no_empty sub result > 1, need nms to merge in overlap area
            merge_result, keep = self.nms(merge_result, scores)
        return merge_result

    def merge_maskrcnn_benchmark_result(self, corners, results, im_scales=None, image_size=None):
        import torch
        from ..deps.mini_maskrcnn_benchmark.mini_maskrcnn_benchmark.structures.boxlist_ops import BoxList

        def result_fmt(result):
            bbox = result.bbox
            labels = result.extra_fields["labels"].reshape(-1, 1).float()
            scores = result.extra_fields["scores"].reshape(-1, 1)
            det_result = torch.cat([bbox, labels, scores], dim=1).detach().cpu().numpy()
            return det_result

        input_BoxList = isinstance(results[0], BoxList)

        if input_BoxList:
            assert im_scales is not None and image_size is not None, ''

        if input_BoxList:
            det_results = []
            for result, im_scale in zip(results, im_scales):
                det_result = result_fmt(result)
                det_result[:, :4] = det_result[:, :4] / np.array([im_scale[1], im_scale[0], im_scale[1], im_scale[0]])
                det_results.append(det_result)
        else:
            det_results = results
        det_results = self.translate_bboxes(corners, det_results)
        if len(det_results) == 0: return []
        _, keep = self.nms(det_results[:, :4], det_results[:, 5])
        det_results = det_results[keep]

        if input_BoxList:
            merge_result = BoxList(torch.Tensor(det_results[:, :4]), image_size, 'xyxy')
            merge_result.add_field("labels", torch.Tensor(det_results[:, 4]))
            merge_result.add_field("scores", torch.Tensor(det_results[:, 5]))
        else:
            merge_result = det_results
        return merge_result

    def translate_bboxes(self, corners, results):
        """
        :param corners: corner of all sub image
        :param results: result of all sub image, results[i] = np.array([[x1, y1, x2, y2,...]...])
        :return:
        """
        merge_result = []
        for corner, result in zip(corners, results):
            if len(result) == 0:
                continue
            result = result.copy()
            result[:, [0, 2]] += corner[0]
            result[:, [1, 3]] += corner[1]
            merge_result.extend(result.tolist())
        merge_result = np.array(merge_result)
        return merge_result

    def nms(self, merge_result, scores):
        from ..deps.mini_maskrcnn_benchmark.mini_maskrcnn_benchmark.layers import nms as _box_nms
        import torch
        if scores is None:
            scores = torch.ones(size=(len(merge_result),))
        if not isinstance(scores, torch.Tensor):
            scores = torch.Tensor(scores)
        merge_result = torch.Tensor(merge_result)
        keep = _box_nms(merge_result, scores, self.nms_th)
        merge_result = merge_result[keep].detach().cpu().numpy()
        return merge_result, keep.detach().cpu().numpy()


def xywh2xyxy(boxes):
    x, y, w, h = boxes.T
    x2, y2 = x + w, y + h
    return np.array([x, y, x2, y2]).T


def xyxy2xywh(boxes):
    x1, y1, x2, y2 = boxes.T
    return np.array([x1, y1, x2 - x1, y2 - y1]).T


class COCOSplitImage(SplitImage):
    """
        if sub_image_dir is not None: (do not need change Dataset in framework)
            generate sub image and save them in sub_image_dir, name them with they left-up corner point,
                xx.jpg-> xx_200_300.jpg mean sub image's left-up corner is (200, 300)
            images: change file_name and so on.
                [{'file_name': 'xx_200_300.jpg', 'height': sub_image_h, 'width': sub_image_w, 'id': sub_image_id, ...}..]
        else: (need change Dataset in framework, add 'corner' crop)
            not generate sub image.
            images: add 'corner' key-value, not changed file_name
                [{'corner': sub_image_corner ,'height': sub_image_h, 'width': sub_image_w, 'id': sub_image_id}..]
        for both condition:
            annotations: [{'id': new_anno_id, 'image_id': sub_image_id, 'area': new_area, 'size': new_size,
                            'bbox': new_bbox, 'segmentation': new_segmentation, ...}..]
    """

    def __init__(self, pieces=None, sub_image_size=None, overlap=(0, 0), anno_keep_overlap=0.7, *args, **kwargs):
        """
        :param pieces:
        :param sub_image_size:
        :param overlap:
        :param anno_keep_overlap:
        :param sub_image_dir:
        """
        super(COCOSplitImage, self).__init__(*args, **kwargs)
        self.pieces = pieces
        self.sub_image_size = sub_image_size
        self.overlap = overlap
        self.anno_keep_overlap = anno_keep_overlap

    def __turn_anno_to_sub_image_anno(self, bboxes, origin_annos, sub_anno_id, sub_image_id):
        """
        :param bboxes: [(x1, y1, x2, y2)]
        :param origin_annos:
        :param start_new_id:
        :param corner_id:
        :return:
        """
        bboxes = self.__turn_ndarray_to_list(xyxy2xywh(bboxes).astype('float32'))
        new_id = sub_anno_id
        annos = []
        for bbox, origin_anno in zip(bboxes, origin_annos):
            anno = deepcopy(origin_anno)
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            anno['bbox'] = bbox
            anno['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            anno['area'] = w * h
            anno['size'] = np.sqrt(anno['area'])
            anno['id'] = new_id
            anno['image_id'] = sub_image_id
            new_id += 1
            annos.append(anno)
        return annos

    def __turn_image_info_to_sub_image_infos(self, corner, image_info, sub_image_id, generate_sub_image: bool):
        corner = self.__turn_ndarray_to_list(corner.astype('int32'))
        new_image_info = deepcopy(image_info)
        new_image_info['id'] = sub_image_id
        new_image_info['width'] = corner[2] - corner[0]
        new_image_info['height'] = corner[3] - corner[1]
        if generate_sub_image:
            f, ext = os.path.splitext(new_image_info['file_name'])
            new_image_info['file_name'] = '{}_{}_{}{}'.format(f, corner[0], corner[1], ext)
        else:
            new_image_info['corner'] = self.__turn_ndarray_to_list(deepcopy(corner))
        return new_image_info

    def __turn_ndarray_to_list(self, v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, (tuple, list)):
            return [self.__turn_ndarray_to_list(e) for e in v]
        return v

    def get_sub_image_corners_and_annos(self, image_info, annos, sub_anno_id, sub_image_id, generate_sub_image):
        if len(annos) == 0:
            boxes = np.array([]).reshape(0, 4)
        else:
            boxes = np.array([anno['bbox'] for anno in annos])
            boxes = xywh2xyxy(boxes)
        image_size = (image_info['width'], image_info['height'])
        corners, sub_annos, sub_keeps = super(COCOSplitImage, self) \
            .get_sub_image_corner_and_boxes(image_size, boxes, self.pieces, self.sub_image_size,
                                            self.overlap, self.anno_keep_overlap)

        sub_image_infos = []
        for i, (corner, sub_anno, sub_keep) in enumerate(zip(corners, sub_annos, sub_keeps)):
            sub_image_info = self.__turn_image_info_to_sub_image_infos(corner, image_info, sub_image_id,
                                                                       generate_sub_image)
            sub_image_infos.append(sub_image_info)
            if len(sub_anno) > 0:
                orgin_anno = [annos[keep_id] for keep_id in sub_keep]
                sub_annos[i] = self.__turn_anno_to_sub_image_anno(sub_anno,
                                                                  orgin_anno, sub_anno_id, sub_image_id)
            else:
                sub_annos[i] = []
            sub_image_id += 1
            sub_anno_id += len(sub_anno)
        return corners, sub_annos, sub_image_infos, sub_anno_id, sub_image_id

    def __load_coco_data_map(self, json_data):
        image_infos, annotations = json_data['images'], json_data['annotations']
        image_ids = [image_info['id'] for image_info in image_infos]
        image_id_to_annos_id = {image_id: [] for image_id in image_ids}
        image_id_to_image_info = {image_info['id']: image_info for image_info in image_infos}
        anno_id_to_anno = {anno['id']: anno for anno in annotations}
        for anno in json_data['annotations']:
            image_id_to_annos_id[anno['image_id']].append(anno['id'])  # an image_id in anno must contained in 'images'
        return image_ids, image_id_to_image_info, anno_id_to_anno, image_id_to_annos_id

    def __save_annotations(self, json_data, src_annotation_path, dst_annotation_path):
        if dst_annotation_path is not None:
            if os.path.isdir(dst_annotation_path):
                f_dir, f_name = os.path.split(src_annotation_path)
                f, ext = os.path.splitext(f_name)
                if self.pieces is not None:
                    pw, ph = self.pieces
                    save_pth = os.path.join(dst_annotation_path, '{}_pw{}_ph{}{}'.format(f, pw, ph, ext))
                else:
                    sw, sh = self.sub_image_size
                    save_pth = os.path.join(dst_annotation_path, '{}_sw{}_sh{}{}'.format(f, sw, sh, ext))
            else:
                save_pth = dst_annotation_path
            print(save_pth, os.path.abspath(save_pth))
            json.dump(json_data, open(save_pth, 'w'), separators=(',', ':'))

    def cut_image_for_coco_json_dataset(self, src_annotation_path, dst_annotation_path=None,
                                        src_image_dir=None, dst_image_dir=None):
        assert dst_image_dir is None or src_image_dir is not None, \
            'sub_image_dir specified, will save sub images to disk, but src_image_dir not given,' \
            ' it needed to load origin image.'
        generate_sub_image = dst_image_dir is not None

        # get json file info
        json_data = json.load(open(src_annotation_path))
        image_ids, image_id_to_image_info, anno_id_to_anno, image_id_to_annos_id = self.__load_coco_data_map(json_data)

        #
        new_annotations, new_image_infos = [], []
        sub_anno_id, sub_image_id = 0, 0
        for i, image_id in enumerate(image_ids):
            # get image_info and annotations for an image
            annos_id = image_id_to_annos_id[image_id]
            image_info = image_id_to_image_info[image_id]
            annos = [anno_id_to_anno[anno_id] for anno_id in annos_id]

            # get sub images' corner, annotations, image_info
            corners, sub_annos, sub_image_infos, sub_anno_id, sub_image_id = self.get_sub_image_corners_and_annos(
                image_info, annos, sub_anno_id, sub_image_id, generate_sub_image)

            # merge sub image infos and annotations
            for sub_anno in sub_annos:
                if len(sub_anno) > 0:
                    new_annotations.extend(sub_anno)
            new_image_infos.extend(sub_image_infos)

            # save sub image to disk.
            if dst_image_dir is not None:
                image = Image.open(image_info['file_name'])
                for corner, sub_image_info in zip(corners, sub_image_infos):
                    sub_image = image.crop(corner)
                    sub_image.save(os.path.join(dst_image_dir, sub_image_info['file_name']))

            # test_cut_image_for_coco_json_dataset(image_info, annos, sub_image_infos, sub_annos, self, corners)
            # if i > 10: break
        json_data['annotations'] = new_annotations
        json_data['old_images'] = json_data['images']
        json_data['images'] = new_image_infos
        self.__save_annotations(json_data, src_annotation_path, dst_annotation_path)
        return json_data


class COCOMergeResult(MergeResult):

    def __turn_det_result(self, bbox, image_id, old_det_result):
        det_result = deepcopy(old_det_result)
        det_result['bbox'] = xyxy2xywh(np.array(bbox)).tolist()
        det_result['image_id'] = image_id
        return det_result

    def __load_coco_data_map(self, corner_gts):
        merge_image_ids = [image_info['id'] for image_info in corner_gts['old_images']]
        image_id_to_image_info = {image_info['id']: image_info for image_info in corner_gts['images']}

        filename_to_merge_image_id = {image_info['file_name']: image_info['id'] for image_info in
                                      corner_gts['old_images']}
        merge_image_id_to_image_ids = {}
        for image_info in corner_gts['images']:
            merge_image_id = filename_to_merge_image_id[image_info['file_name']]
            if merge_image_id not in merge_image_id_to_image_ids:
                merge_image_id_to_image_ids[merge_image_id] = [image_info['id']]
            else:
                merge_image_id_to_image_ids[merge_image_id].append(image_info['id'])
        assert len(merge_image_ids) == len(merge_image_id_to_image_ids)
        return merge_image_ids, image_id_to_image_info, merge_image_id_to_image_ids

    def __load_det_data_map(self, det_data):
        image_id_to_det_boxes = {}
        for det_bbox in det_data:
            if det_bbox['image_id'] not in image_id_to_det_boxes:
                image_id_to_det_boxes[det_bbox['image_id']] = [det_bbox]
            else:
                image_id_to_det_boxes[det_bbox['image_id']].append(det_bbox)
        return image_id_to_det_boxes

    def __save_file(self, json_data, src_path, dst_path):
        if os.path.isdir(dst_path):
            f_dir, f_name = os.path.split(src_path)
            f, ext = os.path.splitext(f_name)
            save_pth = os.path.join(dst_path,
                                    '{}_merge_nms{}{}'.format(f, self.nms_th if self.use_nms else 'None', ext))
        else:
            save_pth = dst_path
        json.dump(json_data, open(save_pth, 'w'), separators=(',', ':'))
        print('[COCOMergeResult]: save file to', os.path.abspath(save_pth))
        return save_pth

    def __call__(self, corner_gt_file_path, src_det_file_path, dst_det_file_path=None, origin_gt_file_path=None):
        det_data = json.load(open(src_det_file_path))
        corner_gts = json.load(open(corner_gt_file_path))
        if origin_gt_file_path is not None:
            origin_gts = json.load(open(origin_gt_file_path))
            if 'old_images' in corner_gts:
                print(f"[COCOMergeResult]: use images id in '{origin_gt_file_path}' as origin image id ,"
                      f" not 'old_images' in corner_gt_file.")
            corner_gts['old_images'] = origin_gts['images']
        else:
            assert 'old_images' in corner_gts, "'old_images' not in corner_gts, origin_gt_file_path must be given."

        # create data map
        merge_image_ids, image_id_to_image_info, merge_image_id_to_image_ids = self.__load_coco_data_map(corner_gts)
        image_id_to_det_boxes = {image_id: [] for image_id in image_id_to_image_info}
        image_id_to_det_boxes.update(self.__load_det_data_map(det_data))

        merge_image_id_to_det_results = {id: [] for id in merge_image_ids}
        merge_image_id_to_corners = {id: [] for id in merge_image_ids}
        for merge_image_id, image_ids in merge_image_id_to_image_ids.items():
            for image_id in image_ids:
                merge_image_id_to_det_results[merge_image_id].append(image_id_to_det_boxes[image_id])
                merge_image_id_to_corners[merge_image_id].append(image_id_to_image_info[image_id]['corner'])

        # merge all det result
        all_merge_det_results = []
        for merge_image_id in tqdm(merge_image_id_to_det_results):
            corners = merge_image_id_to_corners[merge_image_id]
            det_results_list = merge_image_id_to_det_results[merge_image_id]

            # merge all boxes to origin image
            old_det_results, det_bboxes = [], []
            for det_results in det_results_list:
                old_det_results.extend(det_results)
                det_bboxes.append(np.array([xywh2xyxy(np.array(det_result['bbox']))
                                            for det_result in det_results]))
            merge_boxes = self.translate_bboxes(corners, det_bboxes)

            # nms
            if self.use_nms:
                scores = [det_result['score'] for det_result in old_det_results]
                merge_boxes, keeps = self.nms(merge_boxes, scores)
                old_det_results = [old_det_results[keep] for keep in keeps]

            # turn bbox to det_result
            merge_results = []
            for bbox, old_det_result in zip(merge_boxes, old_det_results):
                det_result = self.__turn_det_result(bbox, merge_image_id, old_det_result)
                merge_results.append(det_result)

            all_merge_det_results.extend(merge_results)

        save_pth = None
        if dst_det_file_path is not None:
            save_pth = self.__save_file(all_merge_det_results, src_det_file_path, dst_det_file_path)
        return all_merge_det_results, save_pth


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("type")
    parser.add_argument("--corner_gt", default='')
    parser.add_argument("--corner_det", default='')
    parser.add_argument("--save_merge_det", default='')
    parser.add_argument("--origin_gt", default="")
    parser.add_argument("--merge_nms_th", default=0.5, type=float)
    args = parser.parse_args()

    if args.type == 'merge':
        assert len(args.corner_det) > 0 and len(args.corner_gt) > 0 and len(args.save_merge_det) > 0, \
            "--corner_gt, --corner_det, --save_merge_det must be given."
        if args.merge_nms_th <= 0 or args.merge_nms_th >= 1.0:
            merger = COCOMergeResult(use_nms=False, nms_th=1.0)
        else:
            merger = COCOMergeResult(use_nms=True, nms_th=args.merge_nms_th)
        print("merge args:", merger.__dict__)
        merger(args.corner_gt, args.corner_det, args.save_merge_det,
               None if len(args.origin_gt) == 0 else args.origin_gt)

    # merger(
    #     '/home/hui/dataset/sanya/cocostyle_release/all/rgb/corner_annotations/tiny_all_rgb_test_coco_pw10_ph5.json',
    #     '../../outputs/tiny/scale/coco_pretrain_base4d2_1x/inference/tiny_all_rgb_corner_pw10_ph5_test_coco/bbox.json',
    #     '../../outputs/tiny/scale/coco_pretrain_base4d2_1x/inference/tiny_all_rgb_corner_pw10_ph5_test_coco/'
    # )

    # merger(
    #     '/home/ubuntu/dataset/visDrone/coco_fmt_annotations/corner/'
    #     'VisDrone2018-DET-val-person_corner_w640h640ow100oh100.json',
    #     '/home/ubuntu/github/sparsercnn/outputs/locanet/visdroneperson/visdroneperson_sparsercnn.res50.1000pro/'
    #     '640_stridein3_ADAMW_1x/inference/coco_instances_results.json',
    #     dst_det_file_path='/home/ubuntu/github/sparsercnn/outputs/locanet/visdroneperson/visdroneperson_sparsercnn.res50.1000pro/'
    #     f'640_stridein3_ADAMW_1x/inference/coco_instances_results_merge{merger.nms_th if merger.use_nms else ""}.json',
    #     origin_gt_file_path='/home/ubuntu/dataset/visDrone/coco_fmt_annotations/VisDrone2018-DET-val-person.json'
    # )
