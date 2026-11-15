"""Training-only budget matching; unchanged official engine and frozen eval path."""
import numpy as np
from vendor.cloud_haze import CloudHazeEngine, iou
from .protocol import _check_reflectance, derived_seed, mask_hash, LEVELS
import itertools

VERSION="ruan-budget-matched-training-v1.1"
class BudgetMatchedEngine(CloudHazeEngine):
    @staticmethod
    def _make_mask(seed,shape_hw,coverage,coverage_jitter=0.0):
        # Targets already vary uniformly over 10..70%; disable extra native +/-3% jitter.
        return CloudHazeEngine._make_mask(seed,shape_hw,coverage,coverage_jitter=0.0)

ENGINE=BudgetMatchedEngine(red_idx=2,nir_idx=6)
def augment_sample(s2,sample_id,epoch,seed=20261113,mode="independent",
                   variant="additive",probability=.5):
    x=_check_reflectance(s2)
    if mode not in ("none","independent","overlap") or variant!="additive" or not 0<=probability<=1:
        raise ValueError("Invalid primary training cloud protocol")
    rng=np.random.default_rng(derived_seed(VERSION+"-parameters",seed,int(epoch),int(sample_id)))
    applied=rng.random()<probability
    if mode=="none" or not applied:
        return x.copy(),dict(applied=False,mode=mode,epoch=int(epoch),sample_id=int(sample_id))
    seasons=sorted(rng.choice(4,size=int(rng.integers(2,5)),replace=False).tolist())
    coverage=float(rng.uniform(.1,.7));level=LEVELS[int(rng.integers(3))]
    cloud_seed=derived_seed(VERSION+"-cloud",seed,int(epoch),int(sample_id))
    shift=8 if mode=="overlap" else 0
    while True:
        masks=ENGINE.export_masks(x.shape,seed=cloud_seed,level=level,seasons=seasons,
            cloud_coverage=coverage,overlap=mode=="overlap",overlap_shift=shift)
        pairs=[iou(masks[a]["binary"],masks[b]["binary"]) for a,b in itertools.combinations(seasons,2)]
        pair=min(pairs)
        if mode!="overlap" or pair>=.6 or shift==0:break
        shift//=2
    out,meta=ENGINE.degrade(x,seed=cloud_seed,level=level,variant=variant,seasons=seasons,
        cloud_coverage=coverage,overlap=mode=="overlap",overlap_shift=shift)
    meta.update(applied=True,mode=mode,epoch=int(epoch),sample_id=int(sample_id),
        pair_iou_min=pair,mask_sha256=mask_hash(masks),training_protocol=VERSION,
        coverage_jitter=0.0,effective_max_shift=shift)
    return out,meta
