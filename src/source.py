"""Petrobras 原始训练集的固定版本。下载和训练使用同一校验规则。"""

from hashlib import sha1
from pathlib import Path

COMMIT = "e607a7327cd9ecc00d1244fb6721d5a69a506d9a"
BLOB = "687bac6fd1a06dd9fa48a99a9005e8e39ce0b374"
FILENAME = "20000_well_tests.csv"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / FILENAME
URL = f"https://raw.githubusercontent.com/petrobras/WPRAutoencoders/{COMMIT}/dataset/{FILENAME}"


def verify_csv(raw: bytes) -> str:
    """按 Git blob 的编码规则校验原始字节，以便与上游文件直接比对。"""
    header = f"blob {len(raw)}\0".encode("ascii")
    actual = sha1(header + raw).hexdigest()
    if actual != BLOB:
        raise ValueError(f"Unexpected dataset checksum: {actual} (expected {BLOB})")
    return actual
