import os
import json
import argparse
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader

from src.model import PixelWarriorNetwork
from src.dataset import get_dataloaders
from src.metrics import calculate_psnr, calculate_ssim, calculate_mse
from src.utils import set_seed

def evaluate_model(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    save_json_path: str = None
) -> Dict[str, Any]:
    """
    Evaluates the model on validation data at native resolution (600x400).
    Predictions are constrained to [0.0, 1.0] before computing PSNR, SSIM, and MSE.
    """
    model.eval()
    per_image_results = []
    psnr_list = []
    ssim_list = []
    mse_list = []

    print(f"\n[Evaluation] Evaluating {len(val_loader)} validation images at native 600x400 resolution...")

    with torch.no_grad():
        for idx, (low_img, high_img, fname) in enumerate(val_loader):
            low_img = low_img.to(device)
            high_img = high_img.to(device)

            pred = model(low_img)
            # Constrain predictions to valid [0, 1] range
            pred = torch.clamp(pred, 0.0, 1.0)

            psnr_val = calculate_psnr(pred, high_img, max_val=1.0)
            ssim_val = calculate_ssim(pred, high_img, data_range=1.0)
            mse_val = calculate_mse(pred, high_img)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            mse_list.append(mse_val)

            per_image_results.append({
                "filename": fname[0],
                "psnr": round(psnr_val, 4),
                "ssim": round(ssim_val, 4),
                "mse": round(mse_val, 6)
            })

    mean_psnr = sum(psnr_list) / len(psnr_list)
    mean_ssim = sum(ssim_list) / len(ssim_list)
    mean_mse = sum(mse_list) / len(mse_list)
    min_psnr = min(psnr_list)
    max_psnr = max(psnr_list)

    summary = {
        "num_val_images": len(psnr_list),
        "mean_psnr": round(mean_psnr, 4),
        "mean_ssim": round(mean_ssim, 4),
        "mean_mse": round(mean_mse, 6),
        "psnr_range": {
            "min_psnr": round(min_psnr, 4),
            "max_psnr": round(max_psnr, 4),
            "span_psnr": round(max_psnr - min_psnr, 4)
        },
        "per_image": per_image_results
    }

    print("=" * 60)
    print("VALIDATION EVALUATION REPORT:")
    print(f"  Validation Images: {len(psnr_list)}")
    print(f"  Mean Validation PSNR: {mean_psnr:.2f} dB")
    print(f"  Mean Validation SSIM: {mean_ssim:.4f}")
    print(f"  Validation MSE:       {mean_mse:.6f}")
    print(f"  PSNR Range:           [{min_psnr:.2f} dB, {max_psnr:.2f} dB] (span: {max_psnr - min_psnr:.2f} dB)")
    print("=" * 60)

    if save_json_path:
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[Evaluation] Saved results to: {save_json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate PixelWarrior baseline model at native resolution")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint (.pth)")
    parser.add_argument("--dataset_root", type=str, default="dataset", help="Path to dataset root")
    parser.add_argument("--split_json", type=str, default="splits/split_seed42.json", help="Path to split JSON")
    parser.add_argument("--output_json", type=str, default="experiments/baseline/eval_val.json", help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    model = PixelWarriorNetwork().to(device)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"[Checkpoint] Loading weights from {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("[Checkpoint] No checkpoint provided or found. Evaluating uninitialized baseline.")

    _, val_loader, _, _ = get_dataloaders(args.dataset_root, args.split_json, batch_size=1)
    evaluate_model(model, val_loader, device, save_json_path=args.output_json)


if __name__ == "__main__":
    main()
