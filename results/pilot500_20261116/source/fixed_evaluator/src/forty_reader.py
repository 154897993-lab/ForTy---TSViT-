"""不依赖 TensorFlow 的 ForTy TFRecord 读取器（纯 numpy + 手写 protobuf 解码）。
用法: python forty_reader.py <shard 路径> [最多读几个样本]
"""
import struct, sys, numpy as np

SHAPES = {"s2": (4,128,128,10), "s2_mask": (4,128,128,10),
          "s1_asc": (4,128,128,3), "s1_asc_mask": (4,128,128,3),
          "s1_desc": (4,128,128,3), "s1_desc_mask": (4,128,128,3),
          "elevation": (128,128,3), "climate": (4,1,1,14),
          "segmentation_labels": (128,128), "id": (), "lat": (), "lon": ()}

def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1; r |= (c & 0x7F) << s; s += 7
        if c < 0x80: return r, i

def _fields(b):
    i = 0
    while i < len(b):
        key, i = _varint(b, i); fn, wt = key >> 3, key & 7
        if wt == 2:
            n, i = _varint(b, i); yield fn, b[i:i+n]; i += n
        elif wt == 0:
            v, i = _varint(b, i); yield fn, v
        elif wt == 5: yield fn, b[i:i+4]; i += 4
        elif wt == 1: yield fn, b[i:i+8]; i += 8

def _feature(b):
    for fn, payload in _fields(b):           # 1 bytes_list, 2 float_list, 3 int64_list
        if fn == 2:
            vals = [np.frombuffer(v, "<f4") for _, v in _fields(payload)]   # packed / 单个 float
            return np.concatenate(vals) if vals else np.zeros(0, "f4")
        if fn == 3:
            out = []
            for f2, v in _fields(payload):
                if isinstance(v, int): out.append(v)
                else:
                    j = 0
                    while j < len(v): x, j = _varint(v, j); out.append(x)
            a = np.array(out, dtype=np.uint64).astype(np.int64)
            return a
        if fn == 1:
            return [bytes(v) for _, v in _fields(payload)]
    return None

def parse_example(b):
    out = {}
    for fn, feats in _fields(b):             # Example.features
        for _, entry in _fields(feats):      # map entry
            k = v = None
            for f3, x in _fields(entry):
                if f3 == 1: k = bytes(x).decode()
                else: v = _feature(x)
            out[k] = v
    for k, shp in SHAPES.items():
        if k in out and isinstance(out[k], np.ndarray):
            a = out[k]
            out[k] = a.reshape(shp) if shp else a[0]
    return out

def read_tfrecord(path, limit=None):
    with open(path, "rb") as f:
        n = 0
        while True:
            h = f.read(8)
            if len(h) < 8: return
            ln, = struct.unpack("<Q", h); f.read(4)
            data = f.read(ln); f.read(4)
            yield parse_example(memoryview(data)); n += 1
            if limit and n >= limit: return

if __name__ == "__main__":
    for ex in read_tfrecord(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3):
        print({k: (v.shape, str(v.dtype)) if isinstance(v, np.ndarray) else v for k, v in ex.items()})
