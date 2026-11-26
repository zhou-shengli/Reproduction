"""训练七类试井分类器和各类别的参数反演网络，保存测试预测与误差。"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    r2_score,
)
from torch import nn

from src.data import (
    CLASSES,
    decode_targets,
    dump_json,
    encode_targets,
    fit_scaler,
    load_csv,
    make_features,
    make_split,
    targets_for,
)
from src.models import WellTestANN, WellTestCNN
from src.source import DATA_PATH

ROOT = Path(__file__).resolve().parent


def predict(model: nn.Module, x: torch.Tensor, batch_size: int = 2048) -> np.ndarray:
    """分批推理并返回 CPU 数组，控制评估时的显存占用。"""
    model.eval()
    with torch.no_grad():
        return torch.cat([model(z) for z in x.split(batch_size)]).cpu().numpy()


def metric_rows(truth, pred, names, model_name, evaluation):
    """在原始物理尺度计算每个参数的误差，不裁剪网络输出。"""
    rows = []
    for j, name in enumerate(names):
        err = np.abs(pred[:, j] - truth[:, j])
        row = {
            "model": model_name,
            "evaluation": evaluation,
            "parameter": name,
            "n": len(truth),
            "mae": float(err.mean()),
            "rmse": float(np.sqrt((err**2).mean())),
            "r2": float(r2_score(truth[:, j], pred[:, j])),
        }
        if name == "skin":
            # S 可接近零，百分比相对误差会被放大，因此记录 MAE 及其占参数跨度的比例。
            row["median_absolute_relative_error_pct"] = None
            row["mae_pct_of_parameter_range"] = float(err.mean() / 7 * 100)
        else:
            row["median_absolute_relative_error_pct"] = float(
                np.median(err / np.abs(truth[:, j])) * 100
            )
            row["mae_pct_of_parameter_range"] = None
        rows.append(row)
    return rows


def fit_one(x, target, split, model, args, name, out, weights=None):
    """训练一个网络，按验证损失选择权重并保存完整训练记录。"""
    device = x.device
    y = torch.as_tensor(target, device=device)
    # 分类标签为一维类别编号，反演标签为二维的归一化参数矩阵。
    classification = target.ndim == 1
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 150, 250], gamma=0.1
    )
    weights_tensor = (
        None
        if weights is None
        else torch.as_tensor(weights, dtype=torch.float32, device=device)
    )

    def loss_fn(output, truth):
        if classification:
            return nn.functional.cross_entropy(output, truth)
        return (((output - truth) ** 2) * weights_tensor).sum(1).mean()

    indices = {
        k: torch.as_tensor(v, dtype=torch.long, device=device) for k, v in split.items()
    }
    best_loss, best_state, best_epoch = float("inf"), None, None
    history = []
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        ordered = indices["train"][torch.randperm(len(indices["train"]), device=device)]
        batches = list(ordered.split(args.batch_size))
        # 全连接层的 BatchNorm 在训练时需要至少两个样本。
        if len(batches[-1]) == 1:
            batches[-2] = torch.cat([batches[-2], batches[-1]])
            batches.pop()
        running = 0.0
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[batch]), y[batch])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name}: nonfinite loss at epoch {epoch}")
            loss.backward()
            if args.clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            running += loss.detach().item() * len(batch)
        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(
                model(x[indices["val"]]), y[indices["val"]]
            ).item()
        if not np.isfinite(validation_loss):
            raise FloatingPointError(
                f"{name}: nonfinite validation loss at epoch {epoch}"
            )
        # 用验证损失选择权重。测试数据只参与最终评估。
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / len(ordered),
                "val_loss": validation_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        scheduler.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            elapsed = time.monotonic() - start
            print(
                f"{name:24s} epoch={epoch:3d}/{args.epochs} train={history[-1]['train_loss']:.6f}"
                f" val={validation_loss:.6f} best={best_epoch} seconds={elapsed:.1f}",
                flush=True,
            )
            pd.DataFrame(history).to_csv(out / f"{name}_history.csv", index=False)
    model.load_state_dict(best_state)
    metadata = {
        "name": name,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "epochs_run": args.epochs,
        "seconds": time.monotonic() - start,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_n": len(split["train"]),
        "val_n": len(split["val"]),
        "test_n": len(split["test"]),
    }
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    torch.save(
        {"state_dict": best_state, "metadata": metadata, "config": config},
        out / f"{name}.pt",
    )
    dump_json(out / f"{name}_fit.json", metadata)
    return model, metadata


def train_classifier(df, x, split, build, args, out):
    """用全部训练类别拟合分类器，返回与测试行号同序的预测类别。"""
    test = split["test"]
    labels = df.model.map({s: i for i, s in enumerate(CLASSES)}).to_numpy(
        dtype=np.int64
    )
    clf, _ = fit_one(x, labels, split, build(len(CLASSES)), args, "classifier", out)
    logits = predict(clf, x[test])
    classes_pred = logits.argmax(1)
    class_pred = np.asarray(CLASSES)[classes_pred]
    cls_metrics = classification_report(
        labels[test],
        classes_pred,
        labels=np.arange(len(CLASSES)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    cls_metrics["accuracy"] = float(accuracy_score(labels[test], classes_pred))
    cls_metrics["confusion_matrix_rows_true_columns_pred"] = confusion_matrix(
        labels[test], classes_pred
    ).tolist()
    cls_metrics["classes"] = CLASSES
    dump_json(out / "classification_metrics.json", cls_metrics)
    probs = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
    cls_predictions = pd.DataFrame(
        {
            "row_id": df.row_id.iloc[test].to_numpy(),
            "true_model": df.model.iloc[test].to_numpy(),
            "predicted_model": class_pred,
        }
    )
    for j, cls in enumerate(CLASSES):
        cls_predictions[f"prob_{cls}"] = probs[:, j]
    cls_predictions.to_csv(out / "classification_predictions.csv", index=False)
    print("TEST CLASSIFICATION ACCURACY", cls_metrics["accuracy"], flush=True)
    return class_pred


def train_regressors(df, x, split, build, args, out, class_pred):
    """在固定划分内按类别反演，分别统计已知类别和自动分类后的误差。"""
    test = split["test"]
    all_metrics = []
    global_targets = df[["k", "skin"]].to_numpy()[test]
    routed_pred = np.full((len(test), 2), np.nan)
    for class_id, cls in enumerate(CLASSES):
        names = targets_for(cls, args.target_set)
        local_split = {
            k: idx[df.model.iloc[idx].to_numpy() == cls] for k, idx in split.items()
        }
        y = encode_targets(df, names)
        # 同一类别的样本共享一个反演器，k 与 S 的损失在归一化尺度上加权。
        weights = [args.k_weight] + [1.0] * (len(names) - 1)
        name = f"regressor_{class_id}"
        reg, _ = fit_one(x, y, local_split, build(len(names)), args, name, out, weights)
        idx = local_split["test"]
        pred = decode_targets(predict(reg, x[idx]), names)
        truth = df[names].to_numpy()[idx]
        all_metrics += metric_rows(truth, pred, names, cls, "oracle_class")
        # 以该类训练集的标签中位数为基线，与反演器在相同测试样本上比较。
        baseline = np.broadcast_to(
            df[names].iloc[local_split["train"]].median().to_numpy(), truth.shape
        )
        all_metrics += metric_rows(truth, baseline, names, cls, "train_median_baseline")
        predictions = pd.DataFrame(
            {"row_id": df.row_id.iloc[idx].to_numpy(), "model": cls}
        )
        for j, parameter in enumerate(names):
            predictions[f"true_{parameter}"] = truth[:, j]
            predictions[f"pred_{parameter}"] = pred[:, j]
        predictions.to_csv(out / f"{name}_predictions.csv", index=False)
        # 真实类别用于评估单个反演器；预测类别用于评估整套流程，包含分类错误。
        if class_pred is not None:
            routed_mask = class_pred == cls
            if routed_mask.any():
                routed_pred[routed_mask] = decode_targets(
                    predict(reg, x[test[routed_mask]]), names
                )[:, :2]
    if class_pred is not None:
        if not np.isfinite(routed_pred).all():
            raise ValueError("Unrouted test samples")
        all_metrics += metric_rows(
            global_targets, routed_pred, ["k", "skin"], "all", "predicted_class_routing"
        )
        routed = pd.DataFrame(
            {
                "row_id": df.row_id.iloc[test].to_numpy(),
                "true_model": df.model.iloc[test].to_numpy(),
                "predicted_model": class_pred,
            }
        )
        for j, parameter in enumerate(["k", "skin"]):
            routed[f"true_{parameter}"] = global_targets[:, j]
            routed[f"pred_{parameter}"] = routed_pred[:, j]
        routed.to_csv(out / "routed_regression_predictions.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(out / "regression_metrics.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATA_PATH)
    p.add_argument(
        "--protocol", choices=["physical", "paper_style"], default="physical"
    )
    p.add_argument("--architecture", choices=["cnn", "ann"], default="cnn")
    p.add_argument(
        "--tasks", choices=["both", "classification", "regression"], default="both"
    )
    p.add_argument(
        "--target-set",
        choices=["ks", "ks_storage"],
        default="ks",
        help="ks: predict only permeability and skin in all classes (default)",
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--k-weight", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--no-bn", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if (
        not np.isfinite([args.lr, args.k_weight, args.clip]).all()
        or min(args.lr, args.k_weight) <= 0
        or args.clip < 0
    ):
        p.error("lr and k-weight must be positive and clip must be nonnegative")
    if args.epochs < 1 or args.batch_size < 2 or args.length < 64:
        p.error("epochs >= 1, batch-size >= 2 and length >= 64 are required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    out = (
        args.output
        or ROOT / "results" / f"{args.protocol}_{args.architecture}_{args.target_set}"
    )
    # 每次运行独占一个目录，防止中断后留下的旧权重与新预测混用。
    if args.output is None:
        try:
            out.mkdir(parents=True)
        except FileExistsError:
            out = Path(mkdtemp(prefix=f"{out.name}_", dir=out.parent))
    else:
        if out.exists() and any(out.iterdir()):
            p.error(
                f"Output directory already contains files: {out}. Choose a new --output."
            )
        out.mkdir(parents=True, exist_ok=True)
    args.output = out
    df, audit = load_csv(args.data)
    split = make_split(df, args.seed)
    features = make_features(df, args.protocol, args.length, args.denoise)
    mean, std = fit_scaler(features, split["train"])
    x = torch.as_tensor((features - mean) / std, device=device)
    rows = df[["row_id", "model"]].copy()
    rows["split"] = ""
    for k, idx in split.items():
        rows.loc[idx, "split"] = k
    rows.to_csv(out / "splits.csv", index=False)
    rows.to_csv(out / "rows.csv", index=False)
    np.savez(out / "preprocessing.npz", mean=mean, std=std)
    serializable_args = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    dump_json(out / "config.json", serializable_args)
    dump_json(out / "audit.json", audit)
    dump_json(
        out / "environment.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        },
    )
    factory = WellTestCNN if args.architecture == "cnn" else WellTestANN

    def build(n):
        return factory(features.shape[1], n, args.length, not args.no_bn)

    class_pred = None
    if args.tasks in ["both", "classification"]:
        class_pred = train_classifier(df, x, split, build, args, out)
    if args.tasks in ["both", "regression"]:
        train_regressors(df, x, split, build, args, out, class_pred)
    print("COMPLETED", out, flush=True)
    dump_json(
        out / "COMPLETED.json",
        {
            "tasks": args.tasks,
            "target_set": args.target_set,
            "epochs_per_model": args.epochs,
        },
    )


if __name__ == "__main__":
    main()
