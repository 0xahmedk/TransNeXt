#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List


def read_train_log(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" in obj:
                rows.append(obj)
    rows.sort(key=lambda x: x.get("epoch", -1))
    return rows


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(plots_dir: Path, run_name: str, rows: List[Dict[str, float]]) -> Dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    if not rows:
        return {}

    epochs = [int(r.get("epoch", 0)) for r in rows]
    out: Dict[str, str] = {}

    if any("train_loss" in r for r in rows) or any("test_loss" in r for r in rows):
        plt.figure(figsize=(8, 4.5))
        if any("train_loss" in r for r in rows):
            plt.plot(epochs, [float(r.get("train_loss", float("nan"))) for r in rows], marker="o", label="train_loss")
        if any("test_loss" in r for r in rows):
            plt.plot(epochs, [float(r.get("test_loss", float("nan"))) for r in rows], marker="o", label="test_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{run_name}: Loss Curves")
        plt.grid(alpha=0.3)
        plt.legend()
        p = plots_dir / f"{run_name}_loss.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        out["loss"] = str(p)

    if any("test_acc1" in r for r in rows) or any("test_acc5" in r for r in rows):
        plt.figure(figsize=(8, 4.5))
        if any("test_acc1" in r for r in rows):
            plt.plot(epochs, [float(r.get("test_acc1", float("nan"))) for r in rows], marker="o", label="test_acc1")
        if any("test_acc5" in r for r in rows):
            plt.plot(epochs, [float(r.get("test_acc5", float("nan"))) for r in rows], marker="o", label="test_acc5")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title(f"{run_name}: Validation Accuracy")
        plt.grid(alpha=0.3)
        plt.legend()
        p = plots_dir / f"{run_name}_accuracy.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        out["acc"] = str(p)

    if any("train_lr" in r for r in rows):
        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs, [float(r.get("train_lr", float("nan"))) for r in rows], marker="o", color="tab:green")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.title(f"{run_name}: Learning Rate")
        plt.grid(alpha=0.3)
        p = plots_dir / f"{run_name}_lr.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        out["lr"] = str(p)

    return out


def write_md(report_path: Path, log_path: Path, rows: List[Dict[str, float]], csv_path: Path, plots: Dict[str, str]) -> None:
    report_dir = report_path.parent.resolve()

    def rel(path: Path) -> str:
        return str(path.resolve().relative_to(report_dir)) if path.resolve().is_relative_to(report_dir) else str(path)

    lines: List[str] = []
    lines.append("# TransNeXt Training Report")
    lines.append("")
    lines.append(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"- Source log: {log_path}")
    lines.append(f"- Epoch rows parsed: {len(rows)}")
    lines.append(f"- CSV: {rel(csv_path)}")

    if rows:
        last = rows[-1]
        lines.append("")
        lines.append("## Final Epoch Snapshot")
        lines.append("")
        for key in ["epoch", "train_loss", "test_loss", "test_acc1", "test_acc5", "train_lr", "n_parameters"]:
            if key in last:
                lines.append(f"- {key}: {last[key]}")

        best_acc1 = max((float(r.get("test_acc1", float("-inf"))) for r in rows if "test_acc1" in r), default=None)
        if best_acc1 is not None:
            lines.append(f"- best_test_acc1: {best_acc1:.4f}")

    if plots:
        lines.append("")
        lines.append("## Plots")
        lines.append("")
        if "loss" in plots:
            lines.append(f"![loss](plots/{Path(plots['loss']).name})")
        if "acc" in plots:
            lines.append(f"![acc](plots/{Path(plots['acc']).name})")
        if "lr" in plots:
            lines.append(f"![lr](plots/{Path(plots['lr']).name})")

    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CSV + plots + markdown from training log.txt")
    parser.add_argument("log", help="Path to training log.txt")
    parser.add_argument("--out-dir", default=str(Path("runs") / f"train_report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"Missing log file: {log_path}")

    rows = read_train_log(log_path)
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "epochs.csv"
    write_csv(csv_path, rows)
    plots = make_plots(plots_dir, log_path.parent.name, rows)

    report_path = out_dir / "report.md"
    write_md(report_path, log_path, rows, csv_path, plots)

    print(f"Training report: {report_path}")
    print(f"Epoch CSV: {csv_path}")
    if plots:
        print(f"Plots: {plots_dir}")
    else:
        print("Plots skipped (matplotlib unavailable or no data).")


if __name__ == "__main__":
    main()
