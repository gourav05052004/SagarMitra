from pathlib import Path
from mmcv import Config, ConfigDict
from mmdet.datasets import DATASETS, ClassBalancedDataset
import numpy as np
import cbnet_port
from data_tools import NAMES, MEAN, STD

@DATASETS.register_module()
class CappedClassBalancedDataset(ClassBalancedDataset):
    def __init__(self, dataset, oversample_thr, repeat_cap=2., **kwargs):
        if isinstance(dataset, dict):
            from mmdet.datasets import build_dataset
            dataset=build_dataset(dataset)
        self.repeat_cap = repeat_cap
        super().__init__(dataset, oversample_thr, **kwargs)

    def _get_repeat_factors(self, dataset, repeat_thr):
        return np.minimum(super()._get_repeat_factors(dataset, repeat_thr), self.repeat_cap).tolist()

def make_config(s):
    here = Path(__file__).resolve().parent
    base = Config.fromfile(str(here/'cbnet_port/model_base.py'))
    override = Config.fromfile(str(here/'cbnet_port/model_swin_s.py'))
    base.merge_from_dict(override.to_dict())
    m = base.model
    m.type = 'MaritimeCascade'
    m.backbone.type = 'CBSwinTransformer'
    m.backbone.use_checkpoint = True
    if s.get('train_scope') in ('partial_backbone','detail_head'):
        # Reentrant checkpointing loses gradients for entirely frozen inputs.
        # Only two final Swin blocks train, so retain their activations directly.
        m.backbone.use_checkpoint = False
    m.neck.type = 'CBFPN'
    m.roi_head.pop('mask_roi_extractor',None)
    m.roi_head.pop('mask_head',None)
    for head in m.roi_head.bbox_head:
        head['num_classes'] = 5
        # maritime_class_weights_v1
        if s.get('class_weights') is not None:
            head['loss_cls']['class_weight'] = s['class_weights']
        head['norm_cfg']['type'] = 'BN'  # Same parameter keys; one GPU, no distributed SyncBN.
    m.rpn_head.anchor_generator.scales = [s['anchor_scale']]
    for cfg in (m.train_cfg.rpn_proposal,m.test_cfg.rpn):
        cfg.pop('nms_across_levels',None)
        cfg.pop('nms_post',None)
    m.test_cfg.rcnn = ConfigDict(score_thr=.001,nms=dict(type='nms',iou_threshold=.5),max_per_img=300)
    norm=dict(type='Normalize',mean=MEAN,std=STD,to_rgb=True)
    train_pipeline=[dict(type='LoadImageFromFile'),dict(type='LoadAnnotations',with_bbox=True),
                    dict(type='Resize',img_scale=tuple(s['train_scale']),keep_ratio=True),
                    dict(type='RandomFlip',flip_ratio=s['flip_probability']), norm,
                    dict(type='Pad',size_divisor=32),dict(type='DefaultFormatBundle'),
                    dict(type='Collect',keys=['img','gt_bboxes','gt_labels'])]
    test_pipeline=[dict(type='LoadImageFromFile'),dict(type='MultiScaleFlipAug',
        img_scale=tuple(s['eval_scale']),flip=False,transforms=[
            dict(type='Resize',keep_ratio=True),dict(type='RandomFlip'),norm,
            dict(type='Pad',size_divisor=32),dict(type='ImageToTensor',keys=['img']),
            dict(type='Collect',keys=['img'])])]
    train=dict(type='CocoDataset',ann_file=(Path(s['root'])/'prepared/train.json').as_posix(),
               img_prefix='',classes=NAMES,pipeline=train_pipeline,filter_empty_gt=False)
    if s['repeat_threshold']>0:
        train=dict(type='CappedClassBalancedDataset',dataset=train,oversample_thr=s['repeat_threshold'],
                   repeat_cap=s['repeat_cap'],filter_empty_gt=False)
    val=dict(type='CocoDataset',ann_file=(Path(s['root'])/'prepared/val.json').as_posix(),img_prefix='',
             classes=NAMES,pipeline=test_pipeline,test_mode=True)
    base.data=dict(train=train,val=val,test=val,samples_per_gpu=1,workers_per_gpu=s['workers'])
    custom_keys={'absolute_pos_embed':dict(decay_mult=0.),
        'relative_position_bias_table':dict(decay_mult=0.),'norm':dict(decay_mult=0.)}
    if s.get('backbone_lr_mult', 1.) != 1.:
        custom_keys['backbone']=dict(lr_mult=float(s['backbone_lr_mult']))
    if s.get('quality_head', False):
        custom_keys['fc_quality']=dict(lr_mult=float(s.get('quality_lr_mult', 3.)))
    if s.get('deformable_roi',False):
        custom_keys['conv_offset']=dict(lr_mult=float(s.get('offset_lr_mult',20.)),decay_mult=0.)
    if s.get('train_scope') == 'partial_backbone':
        custom_keys = {}
        # MMCV chooses one longest matching key; combine LR and decay explicitly.
        import torch
        source = torch.load(s['init_checkpoint'], map_location='cpu')['state_dict']
        from train_policy import trainable
        for name in source:
            if trainable(name, 'partial_backbone'):
                no_decay = ('norm' in name or name.endswith('.bias') or
                            'relative_position_bias_table' in name)
                custom_keys[name] = dict(
                    lr_mult=s['backbone_lr_mult'] if name.startswith('backbone.') else 1.,
                    decay_mult=0. if no_decay else 1.)
        del source
    elif s.get('train_scope') == 'detail_head':
        custom_keys = {'detail_project':dict(lr_mult=1.), 'detail_predict':dict(lr_mult=1.)}
    elif s.get('train_scope'):
        custom_keys['reg_refine']=dict(lr_mult=10.)
        custom_keys['neck']=dict(lr_mult=.5)
        for index in (1,2):
            custom_keys[f'roi_head.bbox_head.{index}.shared_convs']=dict(lr_mult=.2)
    base.optimizer=dict(type='AdamW',lr=s['learning_rate'],betas=(.9,.999),weight_decay=.05,foreach=False,
        paramwise_cfg=dict(custom_keys=custom_keys))
    return base
