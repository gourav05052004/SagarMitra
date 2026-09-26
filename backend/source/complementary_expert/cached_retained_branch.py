"""Rebuild the retained tree-score/pixel-box branches from immutable caches.

This is not fresh detector inference. It lets training-side calibration use the
same retained postprocessing as validation rather than an earlier score state.
"""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import torch
import evaluate_complement as e
from portable_tree import predict

PARENT=e.ROOT.parent/'Maritime_CBNet62_QualityEnsemble_backend/runs/quality_001'
PIXEL=e.ROOT.parent/'Maritime_CBNet62_PixelBoundary_backend/runs/pixel_001'


class RetainedBranch:
    def __init__(self):
        torch.set_num_threads(4)
        self.record=e.read(e.ROOT/'retained_pipeline.json')
        self.parent=e.read(PARENT/'manifest.json')
        self.pixel_manifest=e.read(PIXEL/'manifest.json')
        for path,digest in self.pixel_manifest['identity']['settings']['extension_hashes'].items():
            assert e.sha(path)==digest
        assert e.sha(self.record['pixel_checkpoint'])==self.record['pixel_checkpoint_sha256']
        self.tree=e.ROOT/'camera_quality/geometry_portable.json'
        assert e.sha(self.tree)==e.read(e.ROOT/'camera_quality/portable_verification.json')['model_sha256']
        assert e.read(self.tree)['source_sha256']==self.record['tree_model_sha256']
        module_path=PIXEL.parent.parent/'pixel_core.py'
        spec=importlib.util.spec_from_file_location('retained_pixel_inference',module_path)
        self.core=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.core)
        state=torch.load(self.record['pixel_checkpoint'],map_location='cpu')
        assert state['signature']==self.pixel_manifest['signature']
        self.model=self.core.PrecisionHead(state['model']['mean'],state['model']['std']).eval()
        self.model.load_state_dict(state['model'])

    def __call__(self,split,image):
        iid=image['id']
        source=torch.load(PARENT/'cache'/split/f'{iid}.pt',map_location='cpu')
        assert source['signature']==self.parent['signature']
        original=source['rows'];rows=copy.deepcopy(original)
        deltas=predict(self.tree,source['data']['features'].numpy(),source['data']['category'].numpy())
        assert len(deltas)==len(source['indices'])
        for index,delta in zip(source['indices'],deltas):
            row=rows[int(index)];strength=self.record['tree_strengths'].get(str(row['category_id']),0.)
            if strength and delta:
                score=np.clip(row['score'],1e-5,1-1e-5)
                row['score']=float(1/(1+np.exp(-(np.log(score/(1-score))+strength*float(delta)))))
        prepared=torch.load(PIXEL/'prepared'/split/f'{iid}.pt',map_location='cpu')
        profiles=torch.load(PIXEL/'pixel_profiles'/split/f'{iid}.pt',map_location='cpu')
        assert prepared['signature']==profiles['signature']==self.pixel_manifest['signature']
        assert len(prepared['rows'])==len(original)
        assert all(all(a[k]==b[k] for k in ('image_id','category_id','bbox'))
                   for a,b in zip(prepared['rows'],original))
        # The frozen scorer was evaluated in different batches when these two
        # training caches were built. Boxes/identities must be exact; tolerate
        # only sub-micro probability roundoff, never a changed score policy.
        assert all(abs(a['score']-b['score'])<=1e-6 for a,b in zip(prepared['rows'],original))
        stat=Path(image['file_name']).stat()
        assert profiles['image_stamp']==[stat.st_size,stat.st_mtime_ns]
        chosen=profiles['chosen']
        indices=[int(prepared['indices'][int(i)]) for i in chosen]
        assert np.array_equal(chosen,np.array([i for i,j in enumerate(prepared['indices'])
                                               if prepared['rows'][int(j)]['score']>=.05],dtype=np.int64))
        categories=torch.tensor([original[i]['category_id'] for i in indices],dtype=torch.long)
        boxes=torch.tensor([original[i]['bbox'] for i in indices],dtype=torch.float32).reshape(-1,4)
        boxes[:,2:]+=boxes[:,:2]
        # This verified pixel head uses only profiles and categories in forward;
        # its meta/box arguments preserve the common model interface.
        meta=torch.zeros(len(indices),68,dtype=torch.float32)
        values=[]
        with torch.inference_mode():
            for offset in range(0,len(indices),256):
                sl=slice(offset,offset+256)
                values.append(self.model(profiles['profiles'][sl],meta[sl],categories[sl],boxes[sl]).numpy())
        delta=np.concatenate(values) if values else np.empty((0,4),np.float32)
        localized=self.core.apply_boxes(original,indices,delta,self.record['pixel_strengths'],image['width'],image['height'])
        for row,loc in zip(rows,localized):
            row['bbox']=loc['bbox']
        return rows
