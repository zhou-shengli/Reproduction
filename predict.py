"""根据试井曲线识别模型类别，再调用该类别的网络反演物理参数。"""

import argparse
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import CLASSES, KNOWN, decode_targets, make_features, targets_for
from src.models import WellTestANN, WellTestCNN


def load_model(
    run: Path, name: str, channels: int, outputs: int, config: dict
) -> nn.Module:
    """根据训练配置重建网络，并在 CPU 上加载已保存的权重。"""
    network = {"cnn": WellTestCNN, "ann": WellTestANN}[config["architecture"]]
    model = network(channels, outputs, config["length"], not config["no_bn"])
    # 初版检查点在配置中保存了 pathlib 路径。读取为纯路径，支持跨系统加载。
    legacy_paths = [
        (PureWindowsPath, "pathlib.WindowsPath"),
        (PurePosixPath, "pathlib.PosixPath"),
    ]
    with torch.serialization.safe_globals(legacy_paths):
        checkpoint = torch.load(
            run / f"{name}.pt", map_location="cpu", weights_only=True
        )
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def validate_observations(frame: pd.DataFrame) -> None:
    """新样本只需要工程条件与曲线。真实类别、渗透率和表皮系数可省略。"""
    times = [f"t{i}" for i in range(20)]
    signals = [f"{prefix}{i}" for prefix in ("dp", "derp") for i in range(20)]
    required = [*KNOWN, *times, *signals]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    values = frame[required].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(
            "Engineering inputs and log-scale observations must be finite and positive"
        )
    if (np.diff(frame[times].to_numpy(dtype=float), axis=1) <= 0).any():
        raise ValueError("Sampling times must increase within each curve")


@torch.inference_mode()
def predict_batch(
    frame: pd.DataFrame,
    config: dict,
    mean: np.ndarray,
    std: np.ndarray,
    classifier: nn.Module,
    regressors: list[tuple[nn.Module, list[str]]],
) -> pd.DataFrame:
    """按分类器的预测分组推理，保留输入行顺序。"""
    validate_observations(frame)
    features = make_features(
        frame, config["protocol"], config["length"], config["denoise"]
    )
    x = torch.from_numpy((features - mean) / std)
    probabilities = classifier(x).softmax(dim=1).numpy()
    class_ids = probabilities.argmax(axis=1)
    result = pd.DataFrame(
        {
            "predicted_model": np.asarray(CLASSES)[class_ids],
            "max_probability": probabilities.max(axis=1),
        }
    )
    if "row_id" in frame:
        result.insert(0, "row_id", frame["row_id"].to_numpy())
    for index, name in enumerate(CLASSES):
        result[f"prob_{name}"] = probabilities[:, index]
    # 兼容早期三参数井储模型。其他类别的井储列保留为空。
    for parameter in dict.fromkeys(p for _, names in regressors for p in names):
        result[f"pred_{parameter}"] = np.nan
    for index, (model, names) in enumerate(regressors):
        selected = class_ids == index
        if selected.any():
            values = decode_targets(model(x[selected]).numpy(), names)
            result.loc[selected, [f"pred_{p}" for p in names]] = values
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size < 1 or args.output.resolve() == args.input.resolve():
        parser.error(
            "Choose a positive batch size and an output path distinct from the input."
        )
    run = args.run
    config = json.loads((run / "config.json").read_text(encoding="utf8"))
    with np.load(run / "preprocessing.npz") as scale:
        mean, std = scale["mean"], scale["std"]
    channels = mean.shape[1]
    classifier = load_model(run, "classifier", channels, len(CLASSES), config)
    regressors = []
    for index, label in enumerate(CLASSES):
        names = targets_for(label, config.get("target_set", "ks_storage"))
        model = load_model(run, f"regressor_{index}", channels, len(names), config)
        regressors.append((model, names))

    # 分批读取大 CSV。所有批次成功后再替换输出，避免半份预测覆盖已有结果。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    rows = 0
    try:
        with pd.read_csv(args.input, chunksize=args.batch_size) as batches:
            for frame in batches:
                if frame.empty:
                    raise ValueError("Input CSV contains no observations")
                result = predict_batch(frame, config, mean, std, classifier, regressors)
                result.to_csv(temporary, mode="a", index=False, header=rows == 0)
                rows += len(result)
        if rows == 0:
            raise ValueError("Input CSV contains no observations")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {rows} predictions: {args.output}")


if __name__ == "__main__":
    main()
