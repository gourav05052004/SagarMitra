"""Isolated recovery configuration; never writes source images or annotations."""
import numpy as np
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import RandomCrop
from model_config import make_config as original_config
import quality_head  # register the optional final-stage head
import deformable_head
import refinement_head
import detail_roi


@PIPELINES.register_module()
class MixedMaritimeCrop:
    """Random training-only crops mixed with unchanged full frames.

    Crops are selected without using GT locations. Negative crops are retained.
    Truncated boxes with less than 50% retained area or <2 px sides are removed.
    All arrays are loaded from disk anew by the standard dataset pipeline.
    """
    def __init__(self, probability=0.3, size=1536):
        self.probability = probability
        self.crop = RandomCrop(crop_size=(size, size), crop_type='absolute',
                               allow_negative_crop=True, bbox_clip_border=False)

    def __call__(self, results):
        if np.random.random() >= self.probability:
            return results
        results = self.crop(results)
        boxes = results['gt_bboxes']
        h, w = results['img'].shape[:2]
        original_area = np.maximum(boxes[:, 2]-boxes[:, 0], 0)*np.maximum(boxes[:, 3]-boxes[:, 1], 0)
        clipped = boxes.copy()
        clipped[:, 0::2] = np.clip(clipped[:, 0::2], 0, w)
        clipped[:, 1::2] = np.clip(clipped[:, 1::2], 0, h)
        widths, heights = clipped[:, 2]-clipped[:, 0], clipped[:, 3]-clipped[:, 1]
        keep = (widths >= 2) & (heights >= 2) & (widths*heights >= .5*original_area)
        results['gt_bboxes'] = clipped[keep]
        results['gt_labels'] = results['gt_labels'][keep]
        return results


@PIPELINES.register_module()
class ObjectAwareMaritimeCrop:
    """Training-only crop biased toward small labeled objects.

    Remaining crops are random, so negative water/background views remain.
    Source arrays are copied by the standard loader and source files are never
    written. Truncated boxes follow the same retained-area policy as the
    conservative random crop.
    """
    def __init__(self, probability=.3, size=1536, center_probability=.75,
                 min_retained=.5, min_side=2):
        self.probability = float(probability)
        self.size = int(size)
        self.center_probability = float(center_probability)
        self.min_retained = float(min_retained)
        self.min_side = float(min_side)

    @staticmethod
    def _offset(center, crop, extent):
        if crop >= extent:
            return 0
        low = max(0, int(np.ceil(center-crop)))
        high = min(int(np.floor(center)), extent-crop)
        if high < low:
            return int(np.clip(round(center-crop/2), 0, extent-crop))
        return int(np.random.randint(low, high+1))

    def __call__(self, results):
        if np.random.random() >= self.probability:
            return results
        image = results['img']
        height, width = image.shape[:2]
        crop_h, crop_w = min(self.size, height), min(self.size, width)
        boxes = results.get('gt_bboxes', np.empty((0, 4), np.float32))
        if len(boxes) and np.random.random() < self.center_probability:
            area = np.maximum(boxes[:, 2]-boxes[:, 0], 1) * np.maximum(boxes[:, 3]-boxes[:, 1], 1)
            probability = 1/np.sqrt(area)
            probability /= probability.sum()
            target = boxes[np.random.choice(len(boxes), p=probability)]
            left = self._offset((target[0]+target[2])/2, crop_w, width)
            top = self._offset((target[1]+target[3])/2, crop_h, height)
        else:
            left = int(np.random.randint(0, width-crop_w+1))
            top = int(np.random.randint(0, height-crop_h+1))
        results['img'] = image[top:top+crop_h, left:left+crop_w]
        results['img_shape'] = results['img'].shape
        for field in results.get('bbox_fields', []):
            source = results[field]
            if not len(source):
                continue
            original_area = np.maximum(source[:, 2]-source[:, 0], 0) * np.maximum(source[:, 3]-source[:, 1], 0)
            cropped = source.copy()
            cropped[:, 0::2] -= left
            cropped[:, 1::2] -= top
            cropped[:, 0::2] = np.clip(cropped[:, 0::2], 0, crop_w)
            cropped[:, 1::2] = np.clip(cropped[:, 1::2], 0, crop_h)
            widths = cropped[:, 2]-cropped[:, 0]
            heights = cropped[:, 3]-cropped[:, 1]
            keep = ((widths >= self.min_side) & (heights >= self.min_side) &
                    (widths*heights >= self.min_retained*original_area))
            results[field] = cropped[keep]
            if field == 'gt_bboxes' and 'gt_labels' in results:
                results['gt_labels'] = results['gt_labels'][keep]
        return results


def make_config(s):
    cfg = original_config(s)
    stages = s.get('cascade_stages', 3)
    if stages not in (2, 3):
        raise ValueError('Only the original three-stage and explicit two-stage ablation are supported')
    roi = cfg.model.roi_head
    if s.get('detail_head'):
        roi.type = 'DetailCascadeRoIHead'
        roi.detail_max_side = s.get('detail_max_side',192.)
        roi.detail_cap = s.get('detail_cap',2.)
    roi.num_stages = stages
    roi.bbox_head = roi.bbox_head[:stages]
    roi.stage_loss_weights = roi.stage_loss_weights[:stages]
    if s.get('deformable_roi',False):
        for index in s.get('deformable_stages',[0,1,2]):
            roi.bbox_head[index]['type']='RegressionRefinementHead'
            roi.bbox_head[index]['deform_indices']=s.get('deform_indices',[3])
            roi.bbox_head[index]['nwd_weight']=s.get('nwd_weight',0.)
            roi.bbox_head[index]['nwd_constant']=s.get('nwd_constant',16.)
    if s.get('cascade_loss_weights'):
        if len(s['cascade_loss_weights']) != stages:
            raise ValueError('One loss weight is required per Cascade stage')
        roi.stage_loss_weights = list(s['cascade_loss_weights'])
    cfg.model.train_cfg.rcnn = cfg.model.train_cfg.rcnn[:stages]
    roi.bbox_roi_extractor.roi_layer.sampling_ratio = s.get('roi_sampling_ratio', 0)
    if s.get('nms_cfg'):
        cfg.model.test_cfg.rcnn.nms = dict(s['nms_cfg'])
    elif s.get('soft_nms', True):
        cfg.model.test_cfg.rcnn.nms = dict(type='soft_nms', iou_threshold=.5,
                                         method='linear', min_score=.001)
    if s.get('quality_head', False):
        last = roi.bbox_head[-1]
        last['type'] = 'QualityConvFCBBoxHead'
        last['quality_alpha'] = float(s.get('quality_alpha', .35))
        last['quality_loss_weight'] = float(s.get('quality_loss_weight', 4.))
    train = cfg.data.train
    if train.type == 'CappedClassBalancedDataset':
        train = train.dataset
    if s.get('train_annotation'):
        train.ann_file = s['train_annotation'].replace('\\', '/')
    pipeline = train.pipeline
    if s.get('object_crop_probability', 0):
        pipeline.insert(2, dict(type='ObjectAwareMaritimeCrop',
            probability=s['object_crop_probability'], size=s.get('crop_size',1536),
            center_probability=s.get('crop_center_probability',.75),
            min_retained=s.get('crop_min_retained',.5), min_side=2))
    elif s.get('crop_probability', 0):
        pipeline.insert(2, dict(type='MixedMaritimeCrop', probability=s['crop_probability'],
                                size=s.get('crop_size', 1536)))
    for step in pipeline:
        if step['type'] == 'Resize' and s.get('train_scales'):
            step['img_scale'] = [tuple(x) for x in s['train_scales']]
            step['multiscale_mode'] = 'value'
    return cfg
