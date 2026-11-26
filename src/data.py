"""读取 Petrobras 合成试井曲线，构建网络输入并转换回归标签。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .source import COMMIT, verify_csv

CLASSES = [
    "infinite",
    "1 barrier",
    "corner",
    "channel",
    "u shaped",
    "closed",
    "radial storage",
]
KNOWN = ["por", "h", "ct", "Bo", "visc", "q", "rw"]
TARGET_BOUNDS = {"k": (50.0, 1000.0), "skin": (-2.0, 5.0), "stor": (0.01, 5.0)}


def targets_for(name: str, target_set: str = "ks") -> list[str]:
    """默认所有类别预测 k 和 S。旧实验的井储模型还输出 C。"""
    if target_set not in ["ks", "ks_storage"]:
        raise ValueError(f"Unknown target set: {target_set}")
    return (
        ["k", "skin", "stor"]
        if target_set == "ks_storage" and name == "radial storage"
        else ["k", "skin"]
    )


def load_csv(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """校验固定版本的 CSV，保留原始行号，并记录无效样本及重复曲线数量。"""
    path = Path(path)
    raw = path.read_bytes()
    blob = verify_csv(raw)
    df = pd.read_csv(path)
    expected = [
        "model",
        "k",
        "por",
        "h",
        "ct",
        "Bo",
        "visc",
        "q",
        "skin",
        "stor",
        "L1",
        "L2",
        "L3",
        "L4",
        "rw",
    ]
    expected += [f"{p}{i}" for p in ["t", "dp", "derp"] for i in range(20)]
    if list(df.columns) != expected:
        raise ValueError("CSV schema mismatch")
    numeric = df.drop(columns="model").to_numpy(dtype=float)
    time = df[[f"t{i}" for i in range(20)]].to_numpy()
    pressure = df[[f"dp{i}" for i in range(20)]].to_numpy()
    derivative = df[[f"derp{i}" for i in range(20)]].to_numpy()
    valid = (
        np.isfinite(numeric).all(1)
        & (time > 0).all(1)
        & (np.diff(time, axis=1) > 0).all(1)
    )
    valid &= (pressure > 0).all(1) & (derivative > 0).all(1)
    valid &= df.model.isin(CLASSES).to_numpy()
    for k in KNOWN:
        valid &= df[k].to_numpy() > 0
    audit = {
        "source_commit": COMMIT,
        "git_blob_sha1": blob,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "rows_original": len(df),
        "columns": len(df.columns),
        "class_counts_original": df.model.value_counts().to_dict(),
        "nonfinite_rows": int((~np.isfinite(numeric).all(1)).sum()),
        "nonpositive_pressure_rows": int((pressure <= 0).any(1).sum()),
        "nonpositive_derivative_rows": int((derivative <= 0).any(1).sum()),
        "invalid_rows": int((~valid).sum()),
        "duplicate_full_rows": int(df.duplicated().sum()),
    }
    df = df.loc[valid].copy()
    df.insert(0, "row_id", df.index)
    # 按观测曲线去重，防止相同曲线因元数据不同而同时进入训练集和测试集。
    signal_cols = [f"{p}{i}" for p in ["t", "dp", "derp"] for i in range(20)]
    dup = df.duplicated(subset=signal_cols, keep="first")
    audit["duplicate_observed_curves"] = int(dup.sum())
    df = df.loc[~dup].reset_index(drop=True)
    audit["rows_used"] = len(df)
    audit["class_counts_used"] = df.model.value_counts().to_dict()
    return df, audit


def make_split(df: pd.DataFrame, seed: int = 42) -> dict[str, np.ndarray]:
    """按类别分层划分，训练、验证、测试比例依次为 70%、15%、15%。"""
    indices = np.arange(len(df))
    tr, rest = train_test_split(
        indices, test_size=0.3, stratify=df.model, random_state=seed
    )
    va, te = train_test_split(
        rest, test_size=0.5, stratify=df.model.iloc[rest], random_state=seed
    )
    return {k: np.sort(v) for k, v in [("train", tr), ("val", va), ("test", te)]}


def make_features(
    df: pd.DataFrame,
    protocol: str = "physical",
    length: int = 128,
    denoise: bool = False,
) -> np.ndarray:
    """返回形状为 (N, C, length) 的曲线张量。physical 有三个通道，
    paper_style 只使用时间与压力导数，并逐曲线做 min-max 缩放。
    换算只使用已知工程量，k 和 S 保留为监督标签。
    """
    if protocol not in {"physical", "paper_style"}:
        raise ValueError(f"Unknown feature protocol: {protocol}")
    time = df[[f"t{i}" for i in range(20)]].to_numpy(dtype=float)
    dp = df[[f"dp{i}" for i in range(20)]].to_numpy(dtype=float)
    # 上游对 dp / 15.9 求导，乘回该常数才能与压差使用同一尺度。
    der = df[[f"derp{i}" for i in range(20)]].to_numpy(dtype=float) * 15.9
    scale_p = (df.h / (19.03 * df.q * df.Bo * df.visc)).to_numpy()[:, None]
    scale_t = (0.0003484 / (df.por * df.visc * df.ct * df.rw**2)).to_numpy()[:, None]
    log_t, log_p, log_d = (
        np.log10(time * scale_t),
        np.log10(dp * scale_p),
        np.log10(der * scale_p),
    )
    # db4 在 20 个观测点上只支持一级分解，可用层数由原始曲线长度决定。
    if denoise:
        import pywt

        level = min(4, pywt.dwt_max_level(time.shape[1], pywt.Wavelet("db4").dec_len))
        for signal in [log_p, log_d]:
            for i, row in enumerate(signal):
                coeff = pywt.wavedec(row, "db4", level=level, mode="symmetric")
                sigma = np.median(np.abs(coeff[-1] - np.median(coeff[-1]))) / 0.67448975
                threshold = sigma * np.sqrt(2 * np.log(len(row)))
                coeff[1:] = [
                    pywt.threshold(c, threshold, mode="hard") for c in coeff[1:]
                ]
                signal[i] = pywt.waverec(coeff, "db4", mode="symmetric")[: len(row)]
    u = np.linspace(0, 1, length)
    old_u = np.linspace(0, 1, time.shape[1])
    # 上游在对数时间轴等间隔采样。插值只统一网络输入长度，不增加观测点。
    channels = [log_t, log_d] if protocol == "paper_style" else [log_t, log_p, log_d]
    # 对所有曲线复用同一组区间与权重，避免逐行调用插值函数。
    right = np.searchsorted(old_u, u).clip(1, len(old_u) - 1)
    left = right - 1
    fraction = (u - old_u[left]) / (old_u[right] - old_u[left])
    x = np.stack(
        [c[:, left] + fraction * (c[:, right] - c[:, left]) for c in channels],
        axis=1,
    )
    # 外部观测可能不等间隔。此时应按实际对数时间定位，不能按列号插值。
    steps = np.diff(log_t, axis=1)
    irregular = ~np.isclose(steps, steps[:, :1], rtol=1e-7, atol=1e-12).all(axis=1)
    for row in np.flatnonzero(irregular):
        grid = np.linspace(log_t[row, 0], log_t[row, -1], length)
        for channel, signal in enumerate(channels):
            x[row, channel] = np.interp(grid, log_t[row], signal[row])
    if protocol == "paper_style":
        lo = x.min(2, keepdims=True)
        span = x.max(2, keepdims=True) - lo
        x = (x - lo) / np.maximum(span, 1e-12)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite preprocessed features")
    return x.astype(np.float32)


def fit_scaler(
    x: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """仅用训练样本估计每个通道的缩放系数，验证与预测共用这些系数。"""
    mean = x[train_indices].mean(axis=(0, 2), keepdims=True)
    std = np.maximum(x[train_indices].std(axis=(0, 2), keepdims=True), 1e-6)
    return mean, std


def encode_targets(df: pd.DataFrame, names: list[str]) -> np.ndarray:
    """按生成器的已知范围缩放标签，使不同量纲的参数可共同训练。"""
    low = np.array([TARGET_BOUNDS[k][0] for k in names])
    high = np.array([TARGET_BOUNDS[k][1] for k in names])
    return ((df[names].to_numpy() - low) / (high - low)).astype(np.float32)


def decode_targets(y: np.ndarray, names: list[str]) -> np.ndarray:
    """将回归输出还原到原参数尺度，其中渗透率的单位为 mD。"""
    low = np.array([TARGET_BOUNDS[k][0] for k in names])
    high = np.array([TARGET_BOUNDS[k][1] for k in names])
    return y * (high - low) + low


def dump_json(path: str | Path, obj: dict) -> None:
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
