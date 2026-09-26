"""Inference-only numerical HistGradientBoosting trees, without sklearn."""
import json
from pathlib import Path
import numpy as np


def predict_tree(tree, features):
    nodes = np.zeros(len(features), dtype=np.int64)
    leaf = np.asarray(tree['is_leaf'], dtype=bool)
    feature = np.asarray(tree['feature_idx'], dtype=np.int64)
    threshold = np.asarray(tree['num_threshold'], dtype=np.float64)
    left = np.asarray(tree['left'], dtype=np.int64)
    right = np.asarray(tree['right'], dtype=np.int64)
    missing_left = np.asarray(tree['missing_go_to_left'], dtype=bool)
    while True:
        active = np.flatnonzero(~leaf[nodes])
        if not len(active):
            break
        current = nodes[active]
        values = features[active, feature[current]]
        go_left = np.where(np.isnan(values), missing_left[current], values <= threshold[current])
        nodes[active] = np.where(go_left, left[current], right[current])
    return np.asarray(tree['value'], dtype=np.float64)[nodes]


def predict(model_file, features, category):
    document = json.loads(Path(model_file).read_text(encoding='utf-8'))
    features = np.asarray(features, dtype=np.float64)[:, document['columns']]
    category = np.asarray(category)
    result = np.zeros(len(category), dtype=np.float32)
    for name, model in document['models'].items():
        mask = category == int(name)
        selected = features[mask]
        values = np.full(len(selected), model['baseline'], dtype=np.float64)
        for tree in model['trees']:
            # sklearn stores the learning rate in each leaf value already.
            values += predict_tree(tree, selected)
        result[mask] = np.clip(values, -2., 2.)
    return result
