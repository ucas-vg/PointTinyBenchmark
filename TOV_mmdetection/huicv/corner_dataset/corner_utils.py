from pycocotools.coco import COCO
from copy import deepcopy
import numpy as np
import json
from tqdm import tqdm


class CocoAnnoUtil(object):
    @staticmethod
    def filter_small_bbox(annos, size_th=2, area_th=4):
        annotations = []
        for anno in annos:
            bbox = anno['bbox']
            if bbox[-1] <= size_th or bbox[-2] <= size_th:
                continue
            if bbox[-1] * bbox[-2] <= area_th:
                continue
            annotations.append(anno)
        return annotations

    @staticmethod
    def translate(annos, dx, dy, keys=['bbox', 'seg'], ignore_rle=True):
        annos = deepcopy(annos)
        keys = deepcopy(keys)
        if 'seg' in keys:
            for anno in annos:
                segs = []
                segmentation = anno['segmentation']
                if isinstance(segmentation, list):
                    for seg in anno['segmentation']:
                        seg = (np.array(seg).reshape((-1, 2)) + np.array([dx, dy])).reshape((-1,)).tolist()
                        segs.append(seg)
                    anno['segmentation'] = segs
                else:
                    if ignore_rle:
                        pass
                    else:
                        raise NotImplementedError()
            keys.remove('seg')
        for key in keys:
            for anno in annos:
                anno[key][0] += dx
                anno[key][1] += dy
        return annos

    @staticmethod
    def clip(annos, ltrb, keys=['bbox', 'seg']):
        """
        clip to [l, r) and [t, b)
        """
        annos = deepcopy(annos)
        keys = deepcopy(keys)
        if len(annos) == 0: return annos
        keys = deepcopy(keys)
        if 'seg' in keys:
            CocoAnnoUtil.clip_seg(annos, ltrb)
            keys.remove('seg')
        for key in keys:
            CocoAnnoUtil.clip_bbox(annos, ltrb, key)
        return annos

    @staticmethod
    def clip_bbox(annos, ltrb, key='bbox'):
        """
        clip to [l, r) and [t, b)
        """
        cx1, cy1, cx2, cy2 = ltrb
        bboxes = np.array([anno[key] for anno in annos])
        CocoAnnoUtil._bbox_to_xyxy(bboxes)
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(cx1, cx2-1)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(cy1, cy2-1)
        CocoAnnoUtil._bbox_to_xywh(bboxes)
        bboxes = bboxes.tolist()
        for i, anno in enumerate(annos):
            anno[key] = bboxes[i]

    @staticmethod
    def clip_seg(annos, ltrb):
        cx1, cy1, cx2, cy2 = ltrb
        for i, anno in enumerate(annos):
            segs = []
            for seg in anno['segmentation']:
                seg = np.array(seg).reshape((-1, 2))
                seg[:, 0] = seg[:, 0].clip(cx1, cx2-1)
                seg[:, 1] = seg[:, 1].clip(cy1, cy2-1)
                segs.append(seg.reshape((-1,)).tolist())
            anno['segmentation'] = segs

    @staticmethod
    def _bbox_to_xyxy(bboxes):
        bboxes[:, [2, 3]] += bboxes[:, [0, 1]] - 1  # xywh to xyxy

    @staticmethod
    def _bbox_to_xywh(bboxes):
        bboxes[:, [2, 3]] -= bboxes[:, [0, 1]] - 1  # xyxy to xywh

    @staticmethod
    def area(bboxes):
        return (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    @staticmethod
    def crop(annos, corner, area_keep_ratio=0.3, eps=1e-6, size_th=2, area_th=4):
        """
            corner: (l, t, r, b) which mean area of x in [l, r) and y in [t, b)
        """
        annos = deepcopy(annos)
        keep = np.array([True] * len(annos))
        cx1, cy1, cx2, cy2 = corner

        # filter anno by area_keep_ratio
        bboxes = np.array([anno['bbox'] for anno in annos])
        CocoAnnoUtil._bbox_to_xyxy(bboxes)  # xywh to xyxy
        origin_areas = CocoAnnoUtil.area(bboxes)
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(cx1, cx2-1)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(cy1, cy2-1)
        area_ratios = CocoAnnoUtil.area(bboxes) / origin_areas
        for i, anno in enumerate(annos):
            r = area_ratios[i]
            if r < eps:  # out of corner image
                keep[i] = False
            elif r < area_keep_ratio:
                if not anno['ignore']:  # in corner image area < th and is not ignore
                    keep[i] = False
        annos = (np.array(annos)[keep]).tolist()

        if len(annos) > 0:
            ann = annos[0]
            keys = [key for key in ['bbox', 'true_bbox'] if key in ann]
            if 'segmentation' in ann:
                keys.append('seg')
            # recalculate anno bbox and seg in sub image
            annos = CocoAnnoUtil.clip(annos, corner, keys=keys)
            annos = CocoAnnoUtil.translate(annos, -cx1, -cy1, keys=keys, ignore_rle=True)
            # bboxes = np.array([anno['bbox'] for anno in annos2])
            # assert np.sum(bboxes < 0) == 0, (corner, bboxes, annos1, annos2)

            # filter for small bbox
            annos = CocoAnnoUtil.filter_small_bbox(annos, size_th, area_th)
        return annos


class CornerCOCO(object):
    def __init__(self, ann_file):
        """
            ann_file: should be corner annotation file

            self.cornerAnns: dict(corner_ann_id=corner_ann), all annotation in corner image
            self.cornerImgs: dict(corner_img_id=corner_img), all annotation of corner image info.
            self.cornerImgToAnns: dict()

            self.originImgToAnns: dict(), anns here is deepcopy one of corner ann in same origin image, and bbox,
                points, segmentation are translate to origin image. 'image_id' are map to oriImg, but 'id' and others
                still keep as in corner img. so 'id' is import attr to connect origin ann and corner ann.

            self.oriImgs: dict()
            self.
        """
        self.coco = COCO(ann_file)
        self.position_key = self.get_position_keys()

        self.cornerImgs = self.coco.imgs
        self.cornerAnns = self.coco.anns
        self.cornerImgToCornerAnns = self.coco.imgToAnns

        # add corner for not-corner dataset
        self.is_corner = {}
        is_corner_dataset = False
        for i, img_info in self.cornerImgs.items():
            if 'corner' not in img_info:
                img_info['corner'] = [0, 0, img_info["width"], img_info["height"]]
                self.is_corner[i] = False
            else:
                self.is_corner[i] = True
                is_corner_dataset = True

        self.imgNameToCornerImgs = CornerCOCO.group_by_origin_image_name(self.coco.dataset)
        self.oriImgs = {
            (id if is_corner_dataset else img_infos[0]['id']): {
                'id': id if is_corner_dataset else img_infos[0]['id'],
                'file_name': img_infos[0]['file_name'],
                'sub_images': img_infos
            }
            for id, (name, img_infos) in enumerate(self.imgNameToCornerImgs.items())
        }
        self.oriImgToAnns = {}
        self.collect_origin_img_to_annos()
        print("loading annotation over...................")

    def ann_to_corner_ann(self, ann):
        return self.cornerAnns[ann['id']]

    def get_position_keys(self, keys=['bbox', 'true_bbox', 'point']):
        annotations = self.coco.dataset['annotations']
        if len(annotations) > 0:
            keys = list(set(keys))
            ann = annotations[0]
            have_keys = [key for key in keys if key in ann]
            if 'segmentation' in ann:
                have_keys.append('seg')
            return have_keys
        else:
            return []

    @staticmethod
    def group_by_origin_image_name(jd):
        origin_name_to_images = {}
        for image_info in jd['images']:
            file_name = image_info['file_name']
            if file_name not in origin_name_to_images:
                origin_name_to_images[file_name] = [image_info]
            else:
                origin_name_to_images[file_name].append(image_info)
        return origin_name_to_images

    def collect_origin_img_to_annos(self):
        for ori_id, info in self.oriImgs.items():
            self.oriImgToAnns[ori_id] = []
            for img_info in info['sub_images']:
                cx1, cy1, cx2, cy2 = img_info['corner']
                corner_annos = self.cornerImgToCornerAnns[img_info['id']]
                annos = deepcopy(corner_annos)
                annos = CocoAnnoUtil.translate(annos, cx1, cy1, self.position_key, ignore_rle=True)
                for ann in annos:
                    ann['image_id'] = ori_id
                self.oriImgToAnns[ori_id].extend(annos)

    def dump_corner_dataset(self, save_path):
        if len(self.is_corner) > 0:
            for i, is_corner in self.is_corner.items():
                if not is_corner:
                    del self.coco.imgs[i]['corner']

        json.dump(self.coco.dataset, open(save_path, 'w'), separators=(',', ':'))

    @staticmethod
    def parse_results(result_file):
        """
         result: [x1, y1, x2, y2, score, cid]
        """
        results = json.load(open(result_file))
        imgid2results = {}
        for i, res in tqdm(enumerate(results)):
            iid = res['image_id']
            det_res = res['bbox'] + [res['score'], res['category_id']]
            if iid not in imgid2results:
                imgid2results[iid] = [det_res]
            else:
                imgid2results[iid].append(det_res)
            # if (i+1) % 10000 == 0:
            #     print(f'{i}/{len(results)} loading result......')
        imgid2results = {imgid: np.array(results) for imgid, results in imgid2results.items()}
        for imgid, results in imgid2results.items():
            results[:, 2] += results[:, 0]
            results[:, 3] += results[:, 1]
        return imgid2results

    @staticmethod
    def get_imgname2results(result_file, gt_file):
        """
            is origin annotations and result for origin image, not corner
        """
        jd = json.load(open(gt_file))
        imgid2imgname = {}
        for img_info in jd['images']:
            imgid2imgname[img_info['id']] = img_info['file_name']
        imgid2results = CornerCOCO.parse_results(result_file)

        imgname2results = {}
        for iid, img_name in imgid2imgname.items():
            imgname2results[img_name] = imgid2results[iid]
        return imgname2results

    @staticmethod
    def get_imgname2results_corner(result_file, gt_file):
        """
            is corner annotations and result for corner image
        """
        jd = json.load(open(gt_file))
        imgid2imgname = {}
        for img_info in jd['images']:
            imgid2imgname[img_info['id']] = img_info['file_name']
        imgid2results = CornerCOCO.parse_results(result_file)

        imgname2results = {}
        for iid, img_name in imgid2imgname.items():
            imgname2results[img_name] = imgid2results[iid]
        return imgname2results

    @staticmethod
    def group_anns_by_cat_id(anns):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        cid2anns = {}
        for ann in anns:
            cid = ann['category_id']
            if cid not in cid2anns:
                cid2anns[cid] = [ann]
            else:
                cid2anns[cid].append(ann)
        return cid2anns

    @staticmethod
    def group_results_by_cat_id(results):
        """
        results: [[x1, y1, x2, y2, score, cid], ...]
        """
        cid2results = {}
        for r in results:
            if r[5] not in cid2results:
                cid2results[r[5]] = [r]
            else:
                cid2results[r[5]].append(r)
        return {cid: np.array(result) for cid, result in cid2results.items()}
