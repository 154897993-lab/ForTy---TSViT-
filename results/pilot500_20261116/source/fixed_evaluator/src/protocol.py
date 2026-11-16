"""Official cloud_haze integration. Semantic modes avoid ambiguous B1/B2 names."""
import hashlib, itertools
import numpy as np
from vendor.cloud_haze import CloudHazeEngine, iou, __version__
ENGINE = CloudHazeEngine(red_idx=2,nir_idx=6)
VERSION = "paper1-handoff-1.0"
LEVELS = ("light","medium","heavy")
SHAPE = (4,128,128,10)

def derived_seed(namespace, *values):
    s="|".join(map(str,(VERSION,namespace)+values))
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4],"little")

def masks_for(seed, seasons, coverage, level, overlap=False, max_shift=8):
    shift = max_shift if overlap else 0
    while True:
        masks=ENGINE.export_masks(SHAPE,seed=seed,level=level,seasons=seasons,
                                  cloud_coverage=coverage,overlap=overlap,overlap_shift=shift)
        pairs=[iou(masks[a]["binary"],masks[b]["binary"]) for a,b in itertools.combinations(seasons,2)]
        if not overlap or not pairs or min(pairs)>=0.60-1e-12 or shift==0:
            return masks, (min(pairs) if pairs else None), shift
        shift //= 2

def mask_hash(masks):
    h=hashlib.sha256()
    for s in sorted(masks):
        h.update(bytes([s]));h.update(np.asarray(masks[s]["binary"],dtype=np.uint8).tobytes())
    return h.hexdigest()

def _check_reflectance(s2):
    x=np.asarray(s2)
    if x.shape!=SHAPE or not np.issubdtype(x.dtype,np.floating) or not np.isfinite(x).all():
        raise ValueError("Expected finite float [4,128,128,10] reflectance")
    if np.max(np.abs(x))>20:
        raise ValueError("Looks like raw DN: divide by 10000 ONCE before augmentation")
    return x

def augment_sample(s2, sample_id, epoch, seed=20261113, mode="independent",
                   variant="additive", probability=0.5):
    x=_check_reflectance(s2)
    if mode not in ("none","independent","overlap"):
        raise ValueError("mode must be none/independent/overlap")
    if variant not in ("additive","multiplicative") or not 0<=probability<=1:
        raise ValueError("Invalid variant/probability")
    rng=np.random.default_rng(derived_seed("train-parameters",seed,int(epoch),int(sample_id)))
    apply = rng.random()<probability
    if mode=="none" or not apply:
        return x.copy(),dict(applied=False,mode=mode,epoch=int(epoch),sample_id=int(sample_id))
    seasons=sorted(rng.choice(4,size=int(rng.integers(2,5)),replace=False).tolist())
    coverage=float(rng.uniform(.10,.70));level=LEVELS[int(rng.integers(3))]
    cloud_seed=derived_seed("train-cloud",seed,int(epoch),int(sample_id))
    masks,pair,shift=masks_for(cloud_seed,seasons,coverage,level,mode=="overlap")
    out,meta=ENGINE.degrade(x,seed=cloud_seed,level=level,variant=variant,seasons=seasons,
                           cloud_coverage=coverage,overlap=mode=="overlap",overlap_shift=shift)
    meta.update(applied=True,mode=mode,epoch=int(epoch),sample_id=int(sample_id),
                pair_iou_min=pair,mask_sha256=mask_hash(masks))
    return out,meta

def augment_batch(s2, ids, epoch, **kwargs):
    return np.stack([augment_sample(x,int(sid),epoch,**kwargs)[0] for x,sid in zip(s2,ids)])

def build_eval_rows(val_ids, variant="additive", eval_seed=20261116):
    rows=[]
    for sid in sorted(map(int,val_ids)):
        for scene in ("S0","S1","S2","S3"):
            rng=np.random.default_rng(derived_seed("eval-parameters",eval_seed,sid,scene))
            k=0 if scene=="S0" else 1 if scene=="S1" else 2
            seasons=sorted(rng.choice(4,size=k,replace=False).tolist())
            cloud_seed=derived_seed("eval-cloud",eval_seed,sid,scene)
            cov=0.0 if not k else float(rng.uniform(.10,.70))
            level=LEVELS[int(rng.integers(3))]
            masks,pair,shift=({},None,0) if not k else masks_for(cloud_seed,seasons,cov,level,scene=="S3")
            actual=[masks[s]["coverage_actual"] for s in seasons]
            rows.append(dict(sample_id=sid,split="validation",scene=scene,variant=variant,
                             level=level,seasons=";".join(map(str,seasons)),seed=cloud_seed,
                             coverage_target=cov,coverage_actual=";".join(map(str,actual)),
                             overlap_shift=shift,pair_iou="" if pair is None else pair,
                             mask_sha256=mask_hash(masks),engine_version=__version__,protocol_version=VERSION))
    return rows

def apply_eval(s2,row,verify=True):
    x=_check_reflectance(s2)
    if row["engine_version"]!=__version__ or row["protocol_version"]!=VERSION:
        raise ValueError("Engine/protocol version differs from frozen eval manifest")
    seasons=[int(v) for v in row["seasons"].split(";") if v]
    if row["scene"]=="S0":
        if seasons:raise ValueError("S0 must not contain disturbed seasons")
        return x.copy()
    masks,pair,shift=masks_for(int(row["seed"]),seasons,float(row["coverage_target"]),
                              row["level"],row["scene"]=="S3",int(row["overlap_shift"]))
    if verify and mask_hash(masks)!=row["mask_sha256"]:
        raise ValueError("Frozen evaluation mask mismatch")
    out,_=ENGINE.degrade(x,seed=int(row["seed"]),level=row["level"],variant=row["variant"],
                         seasons=seasons,cloud_coverage=float(row["coverage_target"]),
                         overlap=row["scene"]=="S3",overlap_shift=shift)
    return out
