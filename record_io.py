"""Independent TFRecord/Example reader using the official wire schema.

Uncompressed TFRecords only. Requires protobuf, not TensorFlow.
Optional google-crc32c accelerates checksum verification.
"""
import struct
from functools import lru_cache
import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


@lru_cache(maxsize=1)
def example_type():
    file = descriptor_pb2.FileDescriptorProto(name="forty_example.proto", package="forty", syntax="proto3")
    for name, kind in (("BytesList", 12), ("FloatList", 2), ("Int64List", 3)):
        msg = file.message_type.add(name=name)
        field = msg.field.add(name="value", number=1, type=kind, label=3)
        if kind != 12:
            field.options.packed = True
    feature = file.message_type.add(name="Feature")
    feature.oneof_decl.add(name="kind")
    for number, name in enumerate(("bytes_list", "float_list", "int64_list"), 1):
        field = feature.field.add(name=name, number=number, type=11, label=1, oneof_index=0)
        field.type_name = ".forty." + ("BytesList", "FloatList", "Int64List")[number - 1]
    features = file.message_type.add(name="Features")
    entry = features.nested_type.add(name="FeatureEntry")
    entry.options.map_entry = True
    entry.field.add(name="key", number=1, type=9, label=1)
    entry.field.add(name="value", number=2, type=11, label=1, type_name=".forty.Feature")
    features.field.add(name="feature", number=1, type=11, label=3, type_name=".forty.Features.FeatureEntry")
    example = file.message_type.add(name="Example")
    example.field.add(name="features", number=1, type=11, label=1, type_name=".forty.Features")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("forty.Example"))


@lru_cache(maxsize=1)
def _crc_table():
    table = []
    for value in range(256):
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
        table.append(value)
    return table


def crc32c(payload):
    try:
        import google_crc32c
        return google_crc32c.value(payload)
    except ImportError:
        value = 0xFFFFFFFF
        table = _crc_table()
        for byte in payload:
            value = table[(value ^ byte) & 255] ^ (value >> 8)
        return value ^ 0xFFFFFFFF


def masked_crc(payload):
    crc = crc32c(payload)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def read_example(path, byte_offset):
    with open(path, "rb") as stream:
        stream.seek(byte_offset)
        header = stream.read(12)
        if len(header) != 12 or masked_crc(header[:8]) != struct.unpack("<I", header[8:])[0]:
            raise ValueError(f"Invalid TFRecord header at {path}:{byte_offset}")
        length = struct.unpack("<Q", header[:8])[0]
        if not 0 < length <= 64 * 1024 * 1024:
            raise ValueError("Unexpected TFRecord length")
        payload, checksum = stream.read(length), stream.read(4)
    if len(payload) != length or len(checksum) != 4 or masked_crc(payload) != struct.unpack("<I", checksum)[0]:
        raise ValueError(f"Corrupt TFRecord at {path}:{byte_offset}")
    message = example_type()()
    message.ParseFromString(payload)
    features = message.features.feature
    for name, kind in (("s2", "float_list"), ("segmentation_labels", "int64_list"), ("id", "int64_list")):
        if name not in features or features[name].WhichOneof("kind") != kind:
            raise ValueError(f"Missing or incorrectly typed feature: {name}")
    if len(features["id"].int64_list.value) != 1:
        raise ValueError("Expected scalar sample ID")
    return {
        "s2": np.asarray(features["s2"].float_list.value, dtype=np.float32).reshape(4, 128, 128, 10),
        "segmentation_labels": np.asarray(features["segmentation_labels"].int64_list.value, dtype=np.int64).reshape(128, 128),
        "id": int(features["id"].int64_list.value[0]),
    }
