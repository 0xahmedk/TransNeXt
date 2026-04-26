#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TEST_RE = re.compile(
    r"Test:\s+\[\s*(?P<step>\d+)\/(?P<total>\d+)\]\s+eta:\s+(?P<eta>[0-9:]+)\s+"
    r"loss:\s+(?P<loss_batch>[0-9.]+)\s+\((?P<loss_run>[0-9.]+)\)\s+"
    r"acc1:\s+(?P<acc1_batch>[0-9.]+)\s+\((?P<acc1_run>[0-9.]+)\)\s+"
    r"acc5:\s+(?P<acc5_batch>[0-9.]+)\s+\((?P<acc5_run>[0-9.]+)\)\s+"
    r"time:\s+(?P<iter_time>[0-9.]+)\s+data:\s+(?P<data_time>[0-9.]+)"
)
FINAL_RE = re.compile(r"\*\s+Acc@1\s+(?P<acc1>[0-9.]+)\s+Acc@5\s+(?P<acc5>[0-9.]+)\s+loss\s+(?P<loss>[0-9.]+)")
NET_ACC_RE = re.compile(r"Accuracy of the network on the\s+(?P<nimg>\d+)\s+test images:\s+(?P<acc1>[0-9.]+)%")
NS_MODEL_RE = re.compile(r"model='([^']+)'")
NS_DEVICE_RE = re.compile(r"device='([^']+)'")
NS_DATA_RE = re.compile(r"data_path='([^']+)'")
NS_RESUME_RE = re.compile(r"resume='([^']*)'")
NS_BATCH_RE = re.compile(r"batch_size=(\d+)")
NS_EBATCH_RE = re.compile(r"eval_batch_size=(\d+)")
NS_WORKERS_RE = re.compile(r"num_workers=(\d+)")


@dataclass
class RunSummary:
    run_name: str
    log_file: str
    model: str
    device: str
    data_path: str
    checkpoint: str
    batch_size: int
    eval_batch_size: int
    num_workers: int
    total_batches: int
    last_batch: int
    completed: bool
    final_acc1: Optional[float]
    final_acc5: Optional[float]
    final_loss: Optional[float]
    final_images: Optional[int]
    final_reported_acc1: Optional[float]
    mean_iter_time_sec: Optional[float]
    mean_data_time_sec: Optional[float]
    last_running_acc1: Optional[float]
    last_running_acc5: Optional[float]
    last_running_loss: Optional[float]


def parse_namespace_line(line: str) -> Dict[str, object]:
    out: Dict[str, object] = {
        "model": "unknown",
        "device": "unknown",
        "data_path": "unknown",
        "resume": "",
        "batch_size": -1,
        "eval_batch_size": -1,
        "num_workers": -1,
    }
    m = NS_MODEL_RE.search(line)
    if m:
        out["model"] = m.group(1)
    m = NS_DEVICE_RE.search(line)
    if m:
        out["device"] = m.group(1)
    m = NS_DATA_RE.search(line)
    if m:
        out["data_path"] = m.group(1)
    m = NS_RESUME_RE.search(line)
    if m:
        out["resume"] = m.group(1)
    m = NS_BATCH_RE.search(line)
    if m:
        out["batch_size"] = int(m.group(1))
    m = NS_EBATCH_RE.search(line)
    if m:
        out["eval_batch_size"] = int(m.group(1))
    m = NS_WORKERS_RE.search(line)
    if m:
        out["num_workers"] = int(m.group(1))
    return out


def parse_one_log(path: Path) -> Tuple[RunSummary, List[Dict[str, object]]]:
    namespace = {
        "model": "unknown",
        "device": "unknown",
        "data_path": "unknown",
        "resume": "",
        "batch_size": -1,
        "eval_batch_size": -1,
        "num_workers": -1,
    }
    points: List[Dict[str, object]] = []
    final_acc1 = final_acc5 = final_loss = None
    final_images = None
    final_reported_acc1 = None

    with path.open("r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("Namespace("):
                namespace = parse_namespace_line(line)
                continue

            m = TEST_RE.search(line)
            if m:
                points.append(
                    {
                        "step": int(m.group("step")),
                        "total": int(m.group("total")),
                        "eta": m.group("eta"),
                        "loss_batch": float(m.group("loss_batch")),
                        "loss_running": float(m.group("loss_run")),
                        "acc1_batch": float(m.group("acc1_batch")),
                        "acc1_running": float(m.group("acc1_run")),
                        "acc5_batch": float(m.group("acc5_batch")),
                        "acc5_running": float(m.group("acc5_run")),
                        "iter_time": float(m.group("iter_time")),
                        "data_time": float(m.group("data_time")),
                    }
                )
                continue

            m = FINAL_RE.search(line)
            if m:
                final_acc1 = float(m.group("acc1"))
                final_acc5 = float(m.group("acc5"))
                final_loss = float(m.group("loss"))
                continue

            m = NET_ACC_RE.search(line)
            if m:
                final_images = int(m.group("nimg"))
                final_reported_acc1 = float(m.group("acc1"))
                continue

    total_batches = points[-1]["total"] if points else 0
    last_batch = points[-1]["step"] if points else -1
    completed = final_acc1 is not None and final_acc5 is not None and final_loss is not None

    mean_iter = None
    mean_data = None
    if points:
        mean_iter = sum(float(p["iter_time"]) for p in points) / len(points)
        mean_data = sum(float(p["data_time"]) for p in points) / len(points)

    last_running_acc1 = float(points[-1]["acc1_running"]) if points else None
    last_running_acc5 = float(points[-1]["acc5_running"]) if points else None
    last_running_loss = float(points[-1]["loss_running"]) if points else None

    summary = RunSummary(
        run_name=path.stem,
        log_file=str(path),
        model=str(namespace["model"]),
        device=str(namespace["device"]),
        data_path=str(namespace["data_path"]),
        checkpoint=str(namespace["resume"]),
        batch_size=int(namespace["batch_size"]),
        eval_batch_size=int(namespace["eval_batch_size"]),
        num_workers=int(namespace["num_workers"]),
        total_batches=int(total_batches),
        last_batch=int(last_batch),
        completed=completed,
        final_acc1=final_acc1,
        final_acc5=final_acc5,
        final_loss=final_loss,
        final_images=final_images,
        final_reported_acc1=final_reported_acc1,
        mean_iter_time_sec=mean_iter,
        mean_data_time_sec=mean_data,
        last_running_acc1=last_running_acc1,
        last_running_acc5=last_running_acc5,
        last_running_loss=last_running_loss,
    )
    return summary, points


def resolve_logs(inputs: List[str], pattern: str) -> List[Path]:
    out: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_file() and p.suffix == ".log":
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.glob(pattern)))
        else:
            out.extend(sorted(Path().glob(item)))
    unique = sorted(set(out))
    return unique


def write_points_csv(path: Path, points: List[Dict[str, object]]) -> None:
    fieldnames = [
        "step",
        "total",
        "eta",
        "loss_batch",
        "loss_running",
        "acc1_batch",
        "acc1_running",
        "acc5_batch",
        "acc5_running",
        "iter_time",
        "data_time",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in points:
            writer.writerow(row)


def write_summary_csv(path: Path, summaries: List[RunSummary]) -> None:
    fieldnames = list(RunSummary.__annotations__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: getattr(s, k) for k in fieldnames})


def maybe_make_plots(plots_dir: Path, run_name: str, points: List[Dict[str, object]]) -> Dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    if not points:
        return {}

    x = [int(p["step"]) for p in points]
    acc1 = [float(p["acc1_running"]) for p in points]
    acc5 = [float(p["acc5_running"]) for p in points]
    loss = [float(p["loss_running"]) for p in points]
    iter_t = [float(p["iter_time"]) for p in points]
    data_t = [float(p["data_time"]) for p in points]

    outputs = {}

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, acc1, label="Acc@1 (running)")
    plt.plot(x, acc5, label="Acc@5 (running)")
    plt.xlabel("Batch index")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{run_name}: Running Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    p1 = plots_dir / f"{run_name}_running_accuracy.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close()
    outputs["accuracy"] = str(p1)

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, loss, color="tab:red", label="Loss (running)")
    plt.xlabel("Batch index")
    plt.ylabel("Loss")
    plt.title(f"{run_name}: Running Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    p2 = plots_dir / f"{run_name}_running_loss.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=150)
    plt.close()
    outputs["loss"] = str(p2)

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, iter_t, label="Iter time")
    plt.plot(x, data_t, label="Data time")
    plt.xlabel("Batch index")
    plt.ylabel("Seconds")
    plt.title(f"{run_name}: Timing")
    plt.grid(alpha=0.3)
    plt.legend()
    p3 = plots_dir / f"{run_name}_timing.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=150)
    plt.close()
    outputs["timing"] = str(p3)

    return outputs


def write_markdown_report(
    path: Path,
    summaries: List[RunSummary],
    plot_map: Dict[str, Dict[str, str]],
    summary_csv: Path,
    points_dir: Path,
) -> None:
    report_dir = path.parent.resolve()

    def rel_to_report(p: str) -> str:
        try:
            return os.path.relpath(Path(p).resolve(), report_dir)
        except Exception:
            return p

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("# TransNeXt Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append("")
    lines.append("## Scope")
    lines.append("This report summarizes zero-shot evaluation logs for TransNeXt runs.")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Run | Completed | Model | Device | Final Acc@1 | Final Acc@5 | Final Loss | Last Running Acc@1 | Log |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---|")
    for s in summaries:
        final_a1 = f"{s.final_acc1:.3f}" if s.final_acc1 is not None else "-"
        final_a5 = f"{s.final_acc5:.3f}" if s.final_acc5 is not None else "-"
        final_l = f"{s.final_loss:.3f}" if s.final_loss is not None else "-"
        last_a1 = f"{s.last_running_acc1:.3f}" if s.last_running_acc1 is not None else "-"
        lines.append(
            f"| {s.run_name} | {str(s.completed)} | {s.model} | {s.device} | {final_a1} | {final_a5} | {final_l} | {last_a1} | {s.log_file} |"
        )

    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Summary CSV: {rel_to_report(str(summary_csv))}")
    lines.append(f"- Per-run progress CSV folder: {rel_to_report(str(points_dir))}")

    lines.append("")
    lines.append("## Per-Run Details")
    lines.append("")

    for s in summaries:
        lines.append(f"### {s.run_name}")
        lines.append("")
        lines.append(f"- Log file: {rel_to_report(s.log_file)}")
        lines.append(f"- Model: {s.model}")
        lines.append(f"- Device: {s.device}")
        lines.append(f"- Data path: {s.data_path}")
        lines.append(f"- Checkpoint: {s.checkpoint}")
        lines.append(f"- Batch size / eval batch size: {s.batch_size} / {s.eval_batch_size}")
        lines.append(f"- Num workers: {s.num_workers}")
        lines.append(f"- Completed: {s.completed}")
        lines.append(f"- Progress: batch {s.last_batch} / {s.total_batches}")
        if s.final_acc1 is not None:
            lines.append(f"- Final metrics: Acc@1={s.final_acc1:.3f}, Acc@5={s.final_acc5:.3f}, Loss={s.final_loss:.3f}")
        else:
            lines.append("- Final metrics: not found (run likely partial or interrupted)")
        if s.mean_iter_time_sec is not None and s.mean_data_time_sec is not None:
            lines.append(f"- Mean iter/data time (sampled): {s.mean_iter_time_sec:.4f}s / {s.mean_data_time_sec:.4f}s")

        plots = plot_map.get(s.run_name, {})
        if plots:
            lines.append("")
            if "accuracy" in plots:
                lines.append(f"![{s.run_name} accuracy]({rel_to_report(plots['accuracy'])})")
            if "loss" in plots:
                lines.append(f"![{s.run_name} loss]({rel_to_report(plots['loss'])})")
            if "timing" in plots:
                lines.append(f"![{s.run_name} timing]({rel_to_report(plots['timing'])})")
        lines.append("")

    lines.append("## Notes")
    lines.append("- If a run is partial, use last running metrics only as progress indicators.")
    lines.append("- For reproducibility claims, compare only runs with the same checkpoint, mapping, and eval protocol.")
    lines.append("- ImageNet-100 subset results are not directly comparable to full ImageNet-1K paper baselines.")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CSV + plots + markdown report from TransNeXt eval logs")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["runs"],
        help="Log files, directories, or glob patterns (default: runs)",
    )
    parser.add_argument("--pattern", default="*.log", help="Glob pattern inside input directories")
    parser.add_argument(
        "--out-dir",
        default=str(Path("runs") / f"report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        help="Output directory for report artifacts",
    )
    args = parser.parse_args()

    logs = resolve_logs(args.inputs, args.pattern)
    if not logs:
        raise SystemExit("No .log files found for the given inputs.")

    out_dir = Path(args.out_dir)
    csv_dir = out_dir / "csv"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[RunSummary] = []
    plot_map: Dict[str, Dict[str, str]] = {}

    for log_path in logs:
        summary, points = parse_one_log(log_path)
        summaries.append(summary)

        points_csv = csv_dir / f"{summary.run_name}.csv"
        write_points_csv(points_csv, points)

        plots = maybe_make_plots(plots_dir, summary.run_name, points)
        plot_map[summary.run_name] = plots

    summary_csv = out_dir / "summary.csv"
    write_summary_csv(summary_csv, summaries)

    report_md = out_dir / "report.md"
    write_markdown_report(report_md, summaries, plot_map, summary_csv, csv_dir)

    print(f"Report generated: {report_md}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Per-run CSV dir: {csv_dir}")
    if any(plot_map.get(s.run_name) for s in summaries):
        print(f"Plots dir: {plots_dir}")
    else:
        print("Plots were skipped (matplotlib not available or no progress points).")


if __name__ == "__main__":
    main()
