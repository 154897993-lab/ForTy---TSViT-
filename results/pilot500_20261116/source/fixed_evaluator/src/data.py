"""Strict frozen-list lookup and sampled raw CRC checks; never reads official test."""
import csv, struct
from pathlib import Path
import numpy as np
from .forty_reader import parse_example

def manifest(path):
    with open(path,encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    seen=set();locations=set()
    for r in rows:
        sid=int(r['sample_id']);split=r['split'];shard=r['shard']
        if sid in seen:raise ValueError('Duplicate sample ID')
        seen.add(sid)
        if split not in ('train','validation') or '-'+split+'.tfrecord-' not in shard:
            raise ValueError('Official test and split/shard mismatch are prohibited')
        if '/' in shard or '\\' in shard or int(r['byte_offset'])<0:
            raise ValueError('Invalid shard/offset')
        key=(shard,int(r['byte_offset']))
        if key in locations:raise ValueError('Duplicate record location')
        locations.add(key)
        if not (-90<=float(r['lat'])<=90 and -180<=float(r['lon'])<=180):
            raise ValueError('Invalid coordinate')
    return rows

def inventory(rows,raw_root):
    required={r['shard'] for r in rows};found={}
    for p in Path(raw_root).rglob('forty_v1-*.tfrecord-*'):
        if p.is_file() and p.name in required:
            if p.name in found:raise ValueError('Duplicate shard filename locations')
            found[p.name]=str(p)
    return found,sorted(required-set(found))

def validate_sample(x,y,mask):
    if x.shape!=(4,128,128,10) or mask.shape!=x.shape or y.shape!=(128,128):
        raise ValueError('Unexpected shape')
    if not np.isfinite(x).all() or np.max(np.abs(x))>20:raise ValueError('Reflectance invalid')
    if not np.issubdtype(y.dtype,np.integer) or np.any((y<0)|(y>8)):
        raise ValueError('Invalid raw label IDs')
    if np.any((mask!=0)&(mask!=1)):raise ValueError('Invalid validity mask')

def prepared(row,root):
    with np.load(Path(root)/row['split']/(row['sample_id']+'.npz'),allow_pickle=False) as z:
        if int(z['sample_id'])!=int(row['sample_id']):raise ValueError('Prepared ID mismatch')
        x=z['s2'].astype(np.float32);y=z['labels'];mask=z['valid_mask']
    validate_sample(x,y,mask)
    return x,y.astype(np.int64),mask.astype(np.uint8)

def _crc_table():
    out=[]
    for i in range(256):
        v=i
        for _ in range(8):v=(v>>1)^(0x82F63B78 if v&1 else 0)
        out.append(v)
    return out

CRC_TABLE=_crc_table()
def crc32c(data):
    value=0xffffffff
    for byte in data:value=CRC_TABLE[(value^byte)&255]^(value>>8)
    return value^0xffffffff

def masked_crc(data):
    crc=crc32c(data)
    return (((crc>>15)|(crc<<17))+0xa282ead8)&0xffffffff

def raw(row,locations,verify_crc=True):
    with open(locations[row['shard']],'rb') as f:
        f.seek(int(row['byte_offset']));header=f.read(12)
        if len(header)!=12:raise ValueError('Truncated record header')
        size=struct.unpack('<Q',header[:8])[0]
        if not 0<size<256*1024*1024:raise ValueError('Invalid record size')
        body=f.read(size);trailer=f.read(4)
    if len(body)!=size or len(trailer)!=4:raise ValueError('Truncated record')
    if verify_crc and (masked_crc(header[:8])!=struct.unpack('<I',header[8:])[0]
                       or masked_crc(body)!=struct.unpack('<I',trailer)[0]):
        raise ValueError('TFRecord CRC32C mismatch')
    ex=parse_example(memoryview(body))
    if int(ex['id'])!=int(row['sample_id']):raise ValueError('Raw ID/offset mismatch')
    x=np.asarray(ex['s2'],np.float32)/np.float32(10000)
    y=np.asarray(ex['segmentation_labels'],np.int64);mask=np.asarray(ex['s2_mask'],np.uint8)
    validate_sample(x,y,mask)
    return x,y,mask

def load_all(rows,prepared_root=None,raw_root=None):
    if prepared_root:return {int(r['sample_id']):prepared(r,prepared_root) for r in rows}
    if not raw_root:raise ValueError('Provide prepared data or raw shards')
    locations,missing=inventory(rows,raw_root)
    if missing:raise FileNotFoundError('Missing %d required shards; no partial dataset allowed'%len(missing))
    return {int(r['sample_id']):raw(r,locations) for r in rows}
