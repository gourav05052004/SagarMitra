import os
import sys
import json
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "source", "cbnet_runtime"))
sys.path.insert(0, os.path.join(ROOT, "runtime"))

from model_config import make_config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint

def test():
    with open(os.path.join(ROOT, "records", "cbnet_starting_model.json")) as f:
        s = json.load(f)
    s['root'] = ROOT # Override root
    cfg = make_config(s)
    model = build_detector(cfg.model)
    print("Model architecture built.")
    
    ckpt_path = os.path.join(ROOT, "models", "cbnet_primary_update_00200.pth")
    if os.path.exists(ckpt_path):
        checkpoint = load_checkpoint(model, ckpt_path, map_location='cpu')
        print("Checkpoint loaded.")
    else:
        print("Checkpoint file not found:", ckpt_path)

if __name__ == "__main__":
    test()
