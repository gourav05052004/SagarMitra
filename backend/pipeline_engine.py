import os
import sys
import json
import copy
import gc
import torch
import cv2
import numpy as np
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
RUNTIME_DIR = ROOT / "runtime"

bundled_sources = [
    "Maritime_CBNet62_PrecisionBox_backend",
    "allclass_refiners",
    "cbnet_runtime",
    "classsafe",
    "complementary_expert",
    "fast_features",
    "pixel_boundary",
    "ranking"
]

for src in bundled_sources:
    src_path = str(SOURCE_DIR / src)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

# Import lab modules
import ranking_core
import pixel_core
from portable_tree import predict as tree_predict
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
import complement_detector as d
import evaluate_complement as e
import probe_tiles as tiling
import tiny_tile_refinement as tiny

# We use pixel_profiles from runtime as requested by README
import importlib.util
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

pixel_run = load_module('submission_pixel_profiles', str(RUNTIME_DIR / 'pixel_profiles.py'))


class Refiner:
    def __init__(self, config):
        self.core = ranking_core
        self.pixel = pixel_core
        self.tree_predict = tree_predict
        self.pixel_run = pixel_run
        
        # Load scoring normalization
        self.norm = torch.load(ROOT / config['retained_refinement']['scoring_normalization'], map_location='cpu')
        
        # Load relation rank
        state = torch.load(ROOT / config['retained_refinement']['ranking_checkpoint'], map_location='cpu')
        w = state['model']
        self.scorer = ranking_core.ScoringHead(w['meta_mean'], w['meta_std'], state['settings']['score_cap']).eval()
        self.scorer.load_state_dict(w)
        
        # Load pixel boundary
        state = torch.load(ROOT / config['retained_refinement']['pixel_checkpoint'], map_location='cpu')
        self.model = pixel_core.PrecisionHead(state['model']['mean'], state['model']['std']).eval()
        self.model.load_state_dict(state['model'])
        
        # Tree config
        self.tree = str(ROOT / config['retained_refinement']['tree'])
        
        # Fast features
        if torch.cuda.is_available():
            import fast
            fast.native.WEIGHTS = str(ROOT / 'models' / 'resnet18-f37072fd.pth')
            self.fast = fast
            self.encoder = fast.Encoder().cuda().eval()
        else:
            self.fast = None
            self.encoder = None

        # Hardcoded tree/pixel strengths from retained_pipeline.json in records
        retained = json.loads((ROOT / "records" / "retained_pipeline.json").read_text())
        self.tree_strengths = retained.get('tree_strengths', {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0, '5': 1.0})
        self.pixel_strengths = retained.get('pixel_strengths', {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0, '5': 1.0})

    @torch.inference_mode()
    def apply(self, rows, sources, im, bgr):
        indices = self.core.eligible(rows, .001, 1024)
        if not indices:
            return rows

        categories = torch.tensor([rows[j]['category_id'] for j in indices], dtype=torch.long)
        
        # Extract visual fast features
        if self.fast is None:
            visual = torch.zeros((len(indices), 2304)) # Fallback if CPU
        else:
            visual = self.fast.descriptors(self.encoder, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                self.core.boxes(rows)[indices], dict(crop_size=96, context=2., feature_batch=64))
        
        metadata = torch.from_numpy(self.core.metadata(rows, sources, im, indices))
        scores = torch.tensor([rows[j]['score'] for j in indices], dtype=torch.float32)
        
        deltas = []
        for start in range(0, len(indices), 512):
            sl = slice(start, start+512)
            deltas.extend(self.scorer(visual[sl], metadata[sl], categories[sl], scores[sl])[1].numpy())
        
        original = self.core.apply_scores(rows, indices, np.asarray(deltas), {'1':1., '3':1., '4':1.})
        adjusted_scores = torch.tensor([original[j]['score'] for j in indices], dtype=torch.float32)
        
        features = torch.cat([torch.zeros(len(indices), 128),
            ((metadata.float() - self.norm['mean']) / self.norm['std']).clamp(-8, 8),
            torch.logit(adjusted_scores.clamp(1e-5, 1-1e-5))[:, None]], 1)
        
        delta = self.tree_predict(self.tree, features.numpy(), categories.numpy())
        output = self.core.apply_scores(original, indices, delta, self.tree_strengths)
        
        chosen = [j for j in indices if original[j]['score'] >= .05]
        if not chosen:
            return [dict(image_id=im['id'], category_id=r['category_id'], bbox=r['bbox'], score=r['score']) for r in output]

        boxes = torch.tensor([original[j]['bbox'] for j in chosen], dtype=torch.float32).reshape(-1,4)
        boxes[:, 2:] += boxes[:, :2]
        profiles = torch.from_numpy(self.pixel_run.extract_profiles(self.pixel_run.image_fields(bgr), boxes.numpy()))
        cats = torch.tensor([original[j]['category_id'] for j in chosen], dtype=torch.long)
        
        parts = []
        for start in range(0, len(chosen), 256):
            sl = slice(start, start+256)
            parts.append(self.model(profiles[sl], torch.zeros(len(profiles[sl]), 68), cats[sl], boxes[sl]).numpy())
        values = np.concatenate(parts) if parts else np.empty((0, 4), np.float32)
        
        localized = self.pixel.apply_boxes(original, chosen, values, self.pixel_strengths, im['width'], im['height'])
        
        for row, loc in zip(output, localized):
            row['bbox'] = loc['bbox']
            
        return [dict(image_id=im['id'], category_id=r['category_id'], bbox=r['bbox'], score=r['score']) for r in output]


class MaritimePipeline:
    def __init__(self, config_path="pipeline_config.json"):
        self.config_path = ROOT / config_path
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Load models
        print("Loading YOLO...")
        self.yolo = d.YOLO(str(ROOT / self.config['complementary_expert']['checkpoint']))
        
        print("Loading Refiner Ensemble...")
        self.refiner = Refiner(self.config)
        
        print("Loading CBNet...")
        self.cbnet, self.cbnet_settings = self._load_cbnet()

    def _load_cbnet(self):
        from model_config import make_config
        with open(ROOT / "records" / "cbnet_starting_model.json") as f:
            s = json.load(f)['settings']
        s['root'] = str(ROOT) # override root path
        
        cfg = make_config(s)
        model = build_detector(cfg.model)
        checkpoint_path = str(ROOT / self.config['primary_detector']['checkpoint'])
        
        load_checkpoint(model, checkpoint_path, map_location='cpu')
        model.CLASSES = ('swimmer', 'boat', 'jetski', 'life saving appliance', 'buoy')
        model.cfg = cfg
        model.to(self.device)
        model.eval()
        if self.device == 'cuda:0':
            from mmcv.runner import wrap_fp16_model
            wrap_fp16_model(model)
        return model, s

    def predict(self, bgr_image):
        """
        Executes the full 4-stage pipeline.
        """
        im = {'id': 1, 'width': bgr_image.shape[1], 'height': bgr_image.shape[0]}
        
        # We must save the image to a temporary file because CBNet's extra_inference uses cv2.imread
        temp_path = str(ROOT / "temp_inference.jpg")
        cv2.imwrite(temp_path, bgr_image)
        im['file_name'] = temp_path

        try:
            # Stage 1: CBNet Full + Tiled inference
            from worker import extra_inference
            print("Running Stage 1: CBNet...")
            cbnet_rows = extra_inference(self.cbnet, im, self.cbnet_settings, 'tiles')
            
            # The refiner uses sources. Since we don't cache intermediate tiles like the training script, we pass empty dict.
            sources = {} 
            
            # Stage 2: Refiner Ensemble
            print("Running Stage 2: Refiners...")
            refined_rows = self.refiner.apply(cbnet_rows, sources, im, bgr_image)
            
            # Stage 3: YOLO Full inference
            print("Running Stage 3: YOLO...")
            yolo_full_pred = self.yolo.predict(
                temp_path, device='0' if self.device == 'cuda:0' else 'cpu',
                imgsz=self.config['complementary_expert']['imgsz'],
                conf=self.config['complementary_expert']['confidence'],
                iou=self.config['complementary_expert']['nms_iou'],
                max_det=self.config['complementary_expert']['max_det'],
                half=False, augment=False, verbose=False, save=False
            )[0]
            
            yolo_rows = []
            for box, score, cat in zip(yolo_full_pred.boxes.xyxy.cpu().numpy(), yolo_full_pred.boxes.conf.cpu().numpy(), yolo_full_pred.boxes.cls.cpu().numpy()):
                x0, y0, x1, y1 = map(float, box)
                x0, x1 = np.clip([x0, x1], 0, im['width']).tolist()
                y0, y1 = np.clip([y0, y1], 0, im['height']).tolist()
                if x1 > x0 and y1 > y0:
                    yolo_rows.append(dict(image_id=im['id'], category_id=int(cat)+1, bbox=[x0, y0, x1-x0, y1-y0], score=float(score)))
            
            # Fusion
            print("Running Fusion...")
            policy = self.config['expert_fusion'].copy()
            policy['name'] = 'custom'
            fused = e.fuse(refined_rows, yolo_rows, policy)
            
            # Stage 4: YOLO Tiles
            print("Running Stage 4: Tiled Correction...")
            raw = []
            h, w = bgr_image.shape[:2]
            ts = self.config['complementary_expert']['tile_size']
            for top in tiling.starts(h):
                for left in tiling.starts(w):
                    right, bottom = min(w, left+ts), min(h, top+ts)
                    crop = bgr_image[top:bottom, left:right]
                    pred = self.yolo.predict(crop, device='0' if self.device == 'cuda:0' else 'cpu',
                        imgsz=self.config['complementary_expert']['imgsz'], conf=self.config['complementary_expert']['confidence'],
                        iou=self.config['complementary_expert']['nms_iou'], max_det=self.config['complementary_expert']['max_det'],
                        half=False, augment=False, verbose=False, save=False)[0]
                    for box, score, cat in zip(pred.boxes.xyxy.cpu().numpy(), pred.boxes.conf.cpu().numpy(), pred.boxes.cls.cpu().numpy()):
                        if int(cat)+1 not in (1, 4): continue
                        x0, y0, x1, y1 = map(float, box)
                        if ((left > 0 and x0 <= 2) or (top > 0 and y0 <= 2) or (right < w and x1 >= crop.shape[1]-2) or (bottom < h and y1 >= crop.shape[0]-2)): continue
                        if x1 > x0 and y1 > y0: raw.append(dict(image_id=im['id'], category_id=int(cat)+1, bbox=[left+x0, top+y0, x1-x0, y1-y0], score=float(score)))
            
            tiles = tiling.suppress([r for r in yolo_rows if r['category_id'] in (1, 4)] + raw, 'soft')
            final, _ = tiny.refine(fused, tiles, dict(images=[im]))
            
            return final
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def draw_detections(image, detections, config):
    categories = config.get('categories', {})
    annotated = image.copy()
    for det in detections:
        if det['score'] < 0.2: # visualization threshold
            continue
        x, y, w, h = map(int, det['bbox'])
        cat_id = str(det['category_id'])
        score = det['score']
        label = f"{categories.get(cat_id, 'Unknown')} {score:.2f}"
        
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    return annotated

