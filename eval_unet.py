import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

# Reuse training components for consistent preprocessing and metrics
from simple_unet_training import (
    UNet,
    RoadLineDataset,
    find_image_mask_pairs,
    MetricsAccumulator,
    resolve_device,
)


def _overlay_mask(
    image_path: Path, pred_mask: torch.Tensor, true_mask: torch.Tensor | None = None
) -> Image.Image:
    """Create a simple overlay visualization.

    - Red: predicted foreground
    - Green: ground-truth foreground (if provided)
    - Yellow: overlap between pred and gt
    """
    img = Image.open(image_path).convert("RGB").copy()
    w, h = img.size
    pred = pred_mask.to(torch.uint8).cpu().numpy()
    if pred.shape != (h, w):
        # Resize to match original image size for visualization
        pred_img = Image.fromarray((pred * 255).astype("uint8"))
        pred_img = pred_img.resize((w, h), resample=Image.NEAREST)
        pred = (np.array(pred_img) > 127).astype("uint8")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    # Prepare predicted red layer
    red = Image.new("RGBA", (w, h), (255, 0, 0, 80))
    pred_mask_img = Image.fromarray((pred * 255).astype("uint8"))
    overlay.paste(red, (0, 0), pred_mask_img)

    # Optional: add green ground-truth layer
    if true_mask is not None:
        gt = true_mask.to(torch.uint8).cpu().numpy()
        if gt.shape != (h, w):
            gt_img = Image.fromarray((gt * 255).astype("uint8"))
            gt_img = gt_img.resize((w, h), resample=Image.NEAREST)
            gt = (np.array(gt_img) > 127).astype("uint8")
        green = Image.new("RGBA", (w, h), (0, 255, 0, 80))
        gt_mask_img = Image.fromarray((gt * 255).astype("uint8"))
        overlay.paste(green, (0, 0), gt_mask_img)

    composite = Image.alpha_composite(img.convert("RGBA"), overlay)
    return composite.convert("RGB")


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_root: Path,
    device_str: str = "auto",
    batch_size: int = 4,
    num_workers: int = 0,
    visualize: int = 8,
    output_dir: Path | None = None,
) -> None:
    device = resolve_device(device_str)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path} on {device}...")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle both full checkpoint dicts and raw state_dicts
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        training_args = ckpt.get("training_args", {})
        train_stats = ckpt.get("train_stats", None)
        base_channels = int(training_args.get("base_channels", 32))
        image_size = tuple(training_args.get("image_size", (256, 256)))
        threshold = float(training_args.get("val_threshold", 0.75))
        state_dict = ckpt["model_state_dict"]
    else:
        # Fallback to defaults if we only have raw state_dict
        training_args = {}
        train_stats = None
        base_channels = 32
        image_size = (256, 256)
        threshold = 0.75
        state_dict = ckpt

    # Dataset pairs
    test_images = data_root / "test" / "images"
    test_masks = data_root / "test" / "masks"
    pairs: List[Tuple[Path, Path]] = find_image_mask_pairs(test_images, test_masks)
    print(f"Found {len(pairs)} test samples.")

    # Normalization: use training statistics if available
    if train_stats and "image_mean" in train_stats and "image_std" in train_stats:
        mean = train_stats["image_mean"]
        std = train_stats["image_std"]
    else:
        # Compute from test set as a fallback
        from simple_unet_training import compute_dataset_statistics

        stats = compute_dataset_statistics(pairs, image_size)
        mean = stats["image_mean"]
        std = stats["image_std"]
        print("Computed test-set normalization; training stats not found in checkpoint.")

    # Build dataset/loader
    dataset = RoadLineDataset(
        pairs=pairs,
        image_size=image_size,
        augment=False,
        mean=mean,
        std=std,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )

    # Build model and load weights
    model = UNet(in_channels=3, num_classes=2, base_channels=base_channels).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys in state_dict: {sorted(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys in state_dict: {sorted(unexpected)}")
    model.eval()

    metrics = MetricsAccumulator(collect_probabilities=True)
    print("Evaluating on test set...")
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            metrics.update(logits, masks, threshold=threshold)

    summary = metrics.compute()
    print("\n[Test Metrics]")
    print(f"Pixel accuracy    : {summary['pixel_accuracy']:.4f}")
    print(f"Foreground IoU    : {summary['foreground_iou']:.4f}")
    print(f"Foreground Dice   : {summary['foreground_dice']:.4f}")
    print(f"Precision         : {summary['precision']:.4f}")
    print(f"Recall            : {summary['recall']:.4f}")
    print(f"F1                : {summary['f1']:.4f}")
    print(
        f"Boundary F1       : {summary['boundary_f1']:.4f} (P={summary['boundary_precision']:.4f}, R={summary['boundary_recall']:.4f})"
    )
    cm = summary["confusion_matrix"]  # type: ignore[index]
    print(f"Confusion matrix  : TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}")

    # Simple PR curve printout
    curve = metrics.pr_curve(num_thresholds=11)
    if curve:
        formatted = ", ".join(
            f"{thr:.2f}:{prec:.2f}/{rec:.2f}/{f1:.2f}" for thr, prec, rec, f1 in curve
        )
        print(f"PR curve (thr:prec/rec/F1): {formatted}")

    # Optional qualitative visualization
    if visualize > 0:
        vis_out = output_dir if output_dir is not None else checkpoint_path.parent / "test_preds"
        vis_out.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving {visualize} overlay predictions to {vis_out} ...")
        model.eval()
        saved = 0
        with torch.no_grad():
            for idx, (image_path, mask_path) in enumerate(pairs):
                # load and preprocess single image to get prediction
                img = Image.open(image_path).convert("RGB")
                orig_w, orig_h = img.size
                # Build tensor like dataset would
                from torchvision.transforms import functional as TF
                from torchvision.transforms import InterpolationMode

                img_resized = TF.resize(
                    img, image_size, interpolation=InterpolationMode.BILINEAR
                )
                img_tensor = TF.to_tensor(img_resized)
                img_tensor = TF.normalize(img_tensor, mean=mean, std=std)
                img_tensor = img_tensor.unsqueeze(0).to(device)

                logits = model(img_tensor)
                # Probability for foreground class
                probs = torch.softmax(logits, dim=1)[:, 1, ...]
                preds = (probs >= threshold).long().squeeze(0)

                # Ground truth
                gt = Image.open(mask_path).convert("L")
                gt_resized = gt.resize((image_size[1], image_size[0]), resample=Image.NEAREST)
                gt_tensor = torch.from_numpy((np.array(gt_resized) > 0).astype("uint8"))

                overlay = _overlay_mask(image_path, preds, gt_tensor)
                out_path = vis_out / f"sample_{idx:04d}.jpg"
                overlay.save(out_path)
                saved += 1
                if saved >= visualize:
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate U-Net checkpoint on test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path("Model") / "best_unet.pth"),
        help="Path to checkpoint (.pth)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="Dataset root containing train/valid/test",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--visualize", type=int, default=8, help="Number of overlays to save")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional directory to save overlays; defaults to sibling of checkpoint",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    evaluate_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        data_root=Path(args.data_root),
        device_str=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        visualize=args.visualize,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
