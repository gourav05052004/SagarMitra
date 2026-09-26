"""Only train newly introduced detail parameters."""
LEAD = 'roi_head.detail_'


def trainable(name, scope):
    if scope!='detail_head': raise ValueError(scope)
    return name.startswith(LEAD)


def set_training_mode(model):
    # Original detector runs deterministically; eval does not disable autograd.
    model.eval()


def report(model):
    out=dict(detail_trainable=0,original_frozen=0)
    for name,parameter in model.named_parameters():
        if parameter.requires_grad!=trainable(name,'detail_head'):
            raise RuntimeError('Unexpected training scope: '+name)
        out['detail_trainable' if parameter.requires_grad else 'original_frozen']+=parameter.numel()
    assert out['detail_trainable']>0 and out['original_frozen']>0
    return out
