import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve()
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(REPO_ROOT / "classification"))

from transnext import transnext_micro  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(resolution: int):
    return transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model(weights_path: str, resolution: int, pretrain_size: int, fixed_pool_size: int | None, device):
    model = transnext_micro(
        pretrained=False,
        num_classes=37,
        img_size=resolution,
        pretrain_size=pretrain_size,
        fixed_pool_size=fixed_pool_size,
    )

    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)
    model.eval()
    return model


def evaluate_accuracy(model, data_loader, device, max_batches: int):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(data_loader, desc="Evaluating", leave=False)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if max_batches > 0 and batch_idx + 1 >= max_batches:
                break
    return 100.0 * correct / total if total > 0 else 0.0


def enable_attention_capture(model):
    last_block = getattr(model, f"block{model.num_stages}")[-1]
    if hasattr(last_block.attn, "store_attn"):
        last_block.attn.store_attn = True
    return last_block.attn


def denormalize(images: torch.Tensor):
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def save_heatmaps(attn_module, images, resolution, out_dir, max_samples):
    if attn_module.last_attn is None or attn_module.last_hw is None:
        return

    attn = attn_module.last_attn
    attn_map = attn.mean(dim=1).mean(dim=1)
    h, w = attn_module.last_hw
    attn_map = attn_map.reshape(attn_map.shape[0], h, w).unsqueeze(1)
    attn_map = F.interpolate(attn_map, size=(resolution, resolution), mode="bilinear", align_corners=False)
    attn_map = attn_map.squeeze(1)

    images = denormalize(images).cpu()
    attn_map = attn_map.cpu()

    num_samples = min(max_samples, images.shape[0])
    os.makedirs(out_dir, exist_ok=True)

    for idx in range(num_samples):
        heat = attn_map[idx]
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)

        img = images[idx].permute(1, 2, 0).numpy()
        heat_np = heat.numpy()

        plt.figure(figsize=(4, 4))
        plt.imshow(heat_np, cmap="inferno")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(out_dir, f"heatmap_{idx:02d}.png"), dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()

        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.imshow(heat_np, cmap="inferno", alpha=0.45)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(out_dir, f"overlay_{idx:02d}.png"), dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Spatial robustness eval on Oxford-IIIT Pet")
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset root")
    parser.add_argument("--weights", type=str, default="transnext_pets_best.pth", help="Fine-tuned weights")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--use_cpu", action="store_true", help="Use CPU even if GPU is available")
    parser.add_argument("--pretrain_size", type=int, default=224, help="Pretrain size used for relative bias")
    parser.add_argument("--fixed_pool_size", type=int, default=7, help="Fixed pool size for linear inference mode")
    parser.add_argument("--resolutions", type=str, default="224,256,384", help="Comma-separated resolutions")
    parser.add_argument("--heatmap_samples", type=int, default=8, help="Heatmaps per resolution")
    parser.add_argument("--output_dir", type=str, default="runs/pets_spatial_robustness", help="Output directory")
    parser.add_argument("--max_eval_batches", type=int, default=0, help="Limit eval batches (0 = full)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
    print(f"Using device: {device}")

    resolutions = [int(x.strip()) for x in args.resolutions.split(",") if x.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.csv"
    write_header = not results_path.exists()

    with results_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["resolution", "accuracy", "fixed_pool_size"])
        if write_header:
            writer.writeheader()

        for res in resolutions:
            pool_size = args.fixed_pool_size if res != args.pretrain_size else None
            print(f"\n[resolution {res}] fixed_pool_size={pool_size}")
            transform = build_transform(res)
            test_dataset = datasets.OxfordIIITPet(
                root=args.data_dir,
                split="test",
                download=True,
                transform=transform,
            )
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            print(f"Loaded {len(test_dataset)} test samples")

            model = load_model(args.weights, res, args.pretrain_size, pool_size, device)
            attn_module = enable_attention_capture(model)
            attn_module.last_attn = None
            attn_module.last_hw = None

            accuracy = evaluate_accuracy(model, test_loader, device, args.max_eval_batches)
            writer.writerow({
                "resolution": res,
                "accuracy": f"{accuracy:.2f}",
                "fixed_pool_size": pool_size if pool_size is not None else "",
            })
            print(f"Resolution {res}: accuracy {accuracy:.2f}%")
            print(f"Saving heatmaps to {output_dir / f'heatmaps_{res}'}")

            with torch.no_grad():
                for inputs, _ in test_loader:
                    inputs = inputs.to(device)
                    _ = model(inputs)
                    heat_dir = output_dir / f"heatmaps_{res}"
                    save_heatmaps(attn_module, inputs, res, str(heat_dir), args.heatmap_samples)
                    break


if __name__ == "__main__":
    main()
