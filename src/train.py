import os
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.model import PixelWarriorNetwork
from src.dataset import get_dataloaders
from src.metrics import calculate_psnr, calculate_ssim, calculate_mse
from src.utils import set_seed


def train_baseline(
    dataset_root: str = "dataset",
    split_json: str = "splits/split_seed42.json",
    output_dir: str = "experiments/baseline",
    checkpoint_dir: str = "checkpoints",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 0.0001,
    seed: int = 42
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Setup] Running on device: {device}")
    print(f"[Setup] Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {lr}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Dataloaders
    train_loader, val_loader, train_files, val_files = get_dataloaders(
        dataset_root=dataset_root,
        split_json_path=split_json,
        batch_size=batch_size,
        num_workers=0
    )
    print(f"[Dataset] Train samples: {len(train_files)}, Validation samples: {len(val_files)}")

    # 2. Model, Loss, Optimizer, Scheduler
    model = PixelWarriorNetwork().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # AMP scaler if CUDA is available
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    history = {
        "config": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "device": str(device),
            "train_samples": len(train_files),
            "val_samples": len(val_files),
            "model": "PixelWarriorNetwork_Baseline"
        },
        "epochs": []
    }

    best_val_psnr = -float("inf")

    print("\n[Training] Starting baseline training run...")
    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_psnr = 0.0
        num_train_batches = len(train_loader)

        for step, (low_images, high_images, _) in enumerate(train_loader):
            low_images = low_images.to(device)
            high_images = high_images.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(low_images)
                    loss = criterion(outputs, high_images)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(low_images)
                loss = criterion(outputs, high_images)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            # PSNR on clamped predictions for monitoring
            with torch.no_grad():
                clamped_out = torch.clamp(outputs, 0.0, 1.0)
                batch_psnr = calculate_psnr(clamped_out, high_images)
                running_psnr += batch_psnr

            if (step + 1) % 25 == 0 or (step + 1) == num_train_batches:
                print(f"  Epoch [{epoch}/{epochs}] Step [{step+1}/{num_train_batches}] - Loss: {loss.item():.5f} | Batch PSNR: {batch_psnr:.2f} dB")

        epoch_train_loss = running_loss / num_train_batches
        epoch_train_psnr = running_psnr / num_train_batches

        # Validation phase at native resolution
        model.eval()
        val_loss_sum = 0.0
        val_psnr_list = []
        val_ssim_list = []

        with torch.no_grad():
            for low_img, high_img, _ in val_loader:
                low_img = low_img.to(device)
                high_img = high_img.to(device)

                pred = model(low_img)
                pred_c = torch.clamp(pred, 0.0, 1.0)

                loss_val = criterion(pred_c, high_img).item()
                psnr_val = calculate_psnr(pred_c, high_img)
                ssim_val = calculate_ssim(pred_c, high_img)

                val_loss_sum += loss_val
                val_psnr_list.append(psnr_val)
                val_ssim_list.append(ssim_val)

        val_mse = val_loss_sum / len(val_loader)
        val_psnr = sum(val_psnr_list) / len(val_psnr_list)
        val_ssim = sum(val_ssim_list) / len(val_ssim_list)
        min_val_psnr = min(val_psnr_list)
        max_val_psnr = max(val_psnr_list)

        epoch_duration = time.time() - t0
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 5),
            "train_psnr": round(epoch_train_psnr, 2),
            "val_mse": round(val_mse, 6),
            "val_psnr": round(val_psnr, 2),
            "val_ssim": round(val_ssim, 4),
            "val_psnr_min": round(min_val_psnr, 2),
            "val_psnr_max": round(max_val_psnr, 2),
            "duration_sec": round(epoch_duration, 1),
            "lr": round(optimizer.param_groups[0]["lr"], 8)
        }
        history["epochs"].append(epoch_record)

        print(f"\n>>> Epoch [{epoch}/{epochs}] ({epoch_duration:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.5f} | Train PSNR: {epoch_train_psnr:.2f} dB | "
              f"Val MSE: {val_mse:.6f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | "
              f"Range: [{min_val_psnr:.2f}, {max_val_psnr:.2f}] dB")

        # Save latest checkpoint
        latest_ckpt_path = os.path.join(checkpoint_dir, "baseline_latest.pth")
        torch.save(model.state_dict(), latest_ckpt_path)

        # Save best checkpoint
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_ckpt_path = os.path.join(checkpoint_dir, "baseline_best.pth")
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  [*] New best model saved to {best_ckpt_path} (Val PSNR: {val_psnr:.2f} dB)")

    total_training_time = time.time() - start_total_time
    history["total_time_sec"] = round(total_training_time, 1)
    history["best_val_psnr"] = round(best_val_psnr, 2)

    # Save training history
    history_file = os.path.join(output_dir, "train_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Complete] Baseline training finished in {total_training_time:.1f}s.")
    print(f"  Best Val PSNR: {best_val_psnr:.2f} dB")
    print(f"  History logged to: {history_file}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train baseline PixelWarrior network")
    parser.add_argument("--dataset_root", type=str, default="dataset", help="Dataset root dir")
    parser.add_argument("--split_json", type=str, default="splits/split_seed42.json", help="Split JSON path")
    parser.add_argument("--output_dir", type=str, default="experiments/baseline", help="Experiment logs output dir")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoints save dir")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_baseline(
        dataset_root=args.dataset_root,
        split_json=args.split_json,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
