"""Zero-initialized stride-4 detail for P3 box regression only.

The original three classification branches and P4/P5 box branches are kept.
No extra detection anchors are introduced. This is a complementary detector
experiment; the retained CBNet pipeline is never edited.
"""
import copy
from datetime import datetime
import hashlib
from pathlib import Path
import complement_detector as d
import evaluate_complement as e
from ultralytics.nn.modules.head import Detect
from ultralytics.models.yolo.detect.train import DetectionTrainer

torch = d.torch


class RegressionDetailDetect(Detect):
    def forward(self, inputs):
        assert len(inputs) == 4 and self.nl == 3 and not self.end2end
        detail, *features = inputs
        residual = .25 * torch.tanh(self.detail_adapter(detail))
        assert residual.shape == features[0].shape
        outputs = []
        for index, feature in enumerate(features):
            regression_feature = feature + residual if index == 0 else feature
            outputs.append(torch.cat((self.cv2[index](regression_feature), self.cv3[index](feature)), 1))
        if self.training:
            return outputs
        decoded = self._inference(outputs)
        return decoded if self.export else (decoded, outputs)


class RegressionDetailTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        # Preserve the actual custom graph. Re-parsing the old YAML would drop
        # the adapter and silently return a different architecture.
        assert weights is not None and isinstance(weights.model[-1], RegressionDetailDetect)
        model = copy.deepcopy(weights).float()
        if hasattr(model, 'criterion'):
            delattr(model, 'criterion')
        return model


def trainable(name):
    return name.startswith(('model.23.detail_adapter.', 'model.23.cv2.0.'))


def freeze(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(trainable(name))
    frozen_eval(model)


def frozen_eval(model):
    for module in model.model[:23]:
        module.eval()
    head = model.model[-1]
    head.cv3.eval()
    head.cv2[1:].eval()
    head.dfl.eval()


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not trainable(name):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build(checkpoint, output):
    torch.set_num_threads(4)
    torch.manual_seed(92467)
    checkpoint, output = Path(checkpoint), Path(output)
    assert not output.exists(), 'Never overwrite an existing initialized checkpoint.'
    source = d.YOLO(str(checkpoint))
    reference = source.model.cpu().float().eval()
    if hasattr(reference, 'criterion'):
        delattr(reference, 'criterion')
    assert len(reference.model) == 24 and reference.model[-1].f == [16, 19, 22]
    assert reference.model[-1].stride.tolist() == [8., 16., 32.]
    features = []
    hook = reference.model[2].register_forward_hook(lambda module, args, value: features.append(value))
    sample = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        expected = reference(sample)[0]
    hook.remove()
    channels = features[0].shape[1]
    model = copy.deepcopy(reference)
    head = model.model[-1]
    head.__class__ = RegressionDetailDetect
    target_channels = head.cv2[0][0].conv.in_channels
    head.detail_adapter = torch.nn.Sequential(torch.nn.Conv2d(channels, 64, 3, 2, 1, bias=False),
        torch.nn.GroupNorm(8, 64), torch.nn.SiLU(), torch.nn.Conv2d(64, target_channels, 1))
    torch.nn.init.zeros_(head.detail_adapter[-1].weight)
    torch.nn.init.zeros_(head.detail_adapter[-1].bias)
    head.f = [2, 16, 19, 22]
    model.save = sorted(set(model.save) | {2})
    model.eval()
    with torch.no_grad():
        observed = model(sample)[0]
    assert torch.equal(expected, observed), 'Zero adapter must preserve every decoded prediction exactly.'
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=copy.deepcopy(model), ema=None, optimizer=None, epoch=-1, updates=None,
        best_fitness=None, train_args=copy.deepcopy(source.ckpt.get('train_args', {})),
        date=datetime.now().isoformat(), version='8.3.0'), output)
    restored = d.YOLO(str(output)).model.cpu().float().eval()
    with torch.no_grad():
        loaded = restored(sample)[0]
    assert torch.equal(expected, loaded)
    report = dict(source_checkpoint=str(checkpoint.resolve()), source_sha256=e.sha(checkpoint),
        checkpoint=str(output.resolve()), checkpoint_sha256=e.sha(output),
        predictions_exact_at_initialization=True, serialization_exact=True,
        original_parameters=sum(p.numel() for p in reference.parameters()),
        adapted_parameters=sum(p.numel() for p in model.parameters()),
        input_stride=4, regression_stride=8, classification_unchanged=True)
    e.save(output.with_suffix('.json'), report)
    return output, report


def smoke(checkpoint):
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG_DICT
    model = d.YOLO(str(checkpoint)).model.cpu().float()
    if hasattr(model, 'criterion'):
        delattr(model, 'criterion')
    model.args = get_cfg(DEFAULT_CFG_DICT)
    model.train(); freeze(model)
    initial = frozen_digest(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    sample = torch.rand(2, 3, 256, 256)
    batch = dict(img=sample, batch_idx=torch.tensor([0., 0., 1., 1.]),
        cls=torch.tensor([[0.], [3.], [0.], [2.]]),
        bboxes=torch.tensor([[.5,.5,.06,.05],[.25,.25,.04,.04],[.6,.6,.06,.05],[.3,.3,.1,.08]]))
    with torch.no_grad():
        before = [feature[:,64:].clone() for feature in model(sample)]
    weight = model.model[-1].detail_adapter[-1].weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.model[-1].detail_adapter[-1].weight.grad.abs().sum() > 0
        optimizer.step()
    assert frozen_digest(model) == initial
    assert not torch.equal(weight, model.model[-1].detail_adapter[-1].weight)
    with torch.no_grad():
        after = [feature[:,64:] for feature in model(sample)]
    assert all(torch.equal(a,b) for a,b in zip(before,after)), 'Classification logits changed.'
    result = dict(finite_backward=True, adapter_changed=True, frozen_state_unchanged=True,
                  classification_logits_exact=True, loss=float(loss))
    e.save(Path(checkpoint).with_suffix('.smoke.json'), result)
    return result
