"""Raw IDs 0..8; target 0 ignored, prediction 0 on valid truth is FN."""
import numpy as np

NAMES=('natural','planted','tree_crops','other_vegetation','built','water','ice','bare')

def confusion(pred, target, region=None):
    pred=np.asarray(pred); target=np.asarray(target)
    if pred.shape!=target.shape:raise ValueError('Prediction/target shape mismatch')
    if not all(np.issubdtype(x.dtype,np.integer) for x in (pred,target)):
        raise ValueError('Expected integer raw label IDs')
    if any(np.any((x<0)|(x>8)) for x in (pred,target)):
        raise ValueError('Raw IDs must be 0..8')
    valid=target!=0
    if region is not None:
        if np.shape(region)!=target.shape:raise ValueError('Region shape mismatch')
        valid &= np.asarray(region,dtype=bool)
    return np.bincount(9*target[valid]+pred[valid],minlength=81).reshape(9,9)

def summarize(cm):
    cm=np.asarray(cm,dtype=np.float64).copy()
    if cm.shape!=(9,9) or not np.isfinite(cm).all() or (cm<0).any():
        raise ValueError('Invalid confusion matrix')
    cm[0]=0; tp=np.diag(cm)[1:]; support=cm[1:].sum(1); predicted=cm[:,1:].sum(0)
    iou=np.divide(tp,support+predicted-tp,out=np.zeros(8),where=(support+predicted-tp)>0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(8),where=(support+predicted)>0)
    has_data=bool(support.sum())
    return dict(iou=iou.tolist(),f1=f1.tolist(),support=support.astype(int).tolist(),
        valid_pixels=int(support.sum()),absent_class_policy='zero_when_denominator_zero',
        miou=float(iou.mean()) if has_data else None,
        macro_f1=float(f1.mean()) if has_data else None,
        forest_f1=float(f1[:3].mean()) if has_data else None,
        planted_f1=float(f1[1]) if has_data else None)

def dilate(mask,radius):
    out=np.asarray(mask,dtype=bool).copy()
    for _ in range(radius):
        padded=np.pad(out,1,constant_values=False)
        out=np.logical_or.reduce([padded[i:i+out.shape[0],j:j+out.shape[1]]
                                  for i in range(3) for j in range(3)])
    return out

def boundary_band(target,radius=2):
    """4-neighbor GT transitions; Chebyshev radius; exclude nodata neighborhood."""
    y=np.asarray(target)
    if y.ndim!=2 or radius<0:raise ValueError('Expected 2D labels and radius >= 0')
    valid=y!=0; edge=np.zeros(y.shape,dtype=bool)
    horizontal=(y[:,1:]!=y[:,:-1])&valid[:,1:]&valid[:,:-1]
    vertical=(y[1:]!=y[:-1])&valid[1:]&valid[:-1]
    edge[:,1:]|=horizontal; edge[:,:-1]|=horizontal
    edge[1:]|=vertical; edge[:-1]|=vertical
    return dilate(edge,radius)&~dilate(~valid,radius)&valid
