"""
Kaggle Multi-GPU (2x T4 DDP) Training Entry Point for PixelWarrior Canonical Baseline.

Multi-GPU Specification:
- Platform: Kaggle NVIDIA T4 x 2
- Framework: PyTorch DistributedDataParallel (DDP) with NCCL backend
- World Size: 2 GPUs (cuda:0, cuda:1)
- Per-GPU Batch Size: 2
- Effective Global Batch Size: 2 * 2 = 4 (Exact match to audited baseline!)
- Learning Rate: 1e-4 (Unchanged, gradient dynamics preserved)
- Model: Exact baseline PixelWarriorNetwork (31.04M params)
- Seed: 42 (Deterministic per-rank seeding)
- Split: Fixed persisted 435 train / 50 val split
- Loss: MSELoss
- Scheduler: CosineAnnealingLR (T_max=15, eta_min=1e-6)
- Metrics: Clamped PSNR [0, 1] (MAX=1.0) and SSIM at native 600x400 resolution
"""

import os
import sys
import json
import time
import math
import argparse
import glob
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# Import model and metric definitions
from src.model import PixelWarriorNetwork
from src.metrics import calculate_psnr, calculate_ssim, calculate_mse
from src.utils import set_seed


def auto_detect_dataset_root(hint: str = None) -> str:
    """
    Intelligently locates the dataset root containing 'low' and 'high' directories.
    """
    candidates = []
    if hint:
        candidates.append(hint)

    kaggle_patterns = [
        "/kaggle/input/**/dataset",
        "/kaggle/input/**/dataset00",
        "/kaggle/input/**/augmented_Train",
        "/kaggle/input/**/low",
        "/kaggle/input/*",
        "./dataset",
        "../dataset"
    ]
    for pat in kaggle_patterns:
        for match in glob.glob(pat, recursive=True):
            candidates.append(match)

    for c in candidates:
        if os.path.isdir(c):
            if os.path.isdir(os.path.join(c, "low")) and os.path.isdir(os.path.join(c, "high")):
                return c
            parent = os.path.dirname(c)
            if os.path.isdir(os.path.join(parent, "low")) and os.path.isdir(os.path.join(parent, "high")):
                return parent

    raise FileNotFoundError(
        "Could not automatically locate dataset with 'low' and 'high' directories. "
        "Please provide --dataset_root /path/to/dataset"
    )


class LowLightDataset(Dataset):
    def __init__(self, dataset_root: str, file_list: List[str]):
        self.low_dir = os.path.join(dataset_root, "low")
        self.high_dir = os.path.join(dataset_root, "high")
        self.file_list = sorted(file_list)
        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        fname = self.file_list[idx]
        low_img = Image.open(os.path.join(self.low_dir, fname)).convert("RGB")
        high_img = Image.open(os.path.join(self.high_dir, fname)).convert("RGB")
        return self.transform(low_img), self.transform(high_img), fname


def setup_ddp() -> Tuple[int, int, int, torch.device]:
    """
    Initializes DDP process group using NCCL backend.
    Supports both `torchrun` and `mp.spawn`.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if torch.cuda.is_available() and world_size > 1:
        backend = "nccl" if sys.platform != "win32" else "gloo"
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        try:
            dist.init_process_group(backend=backend, init_method="env://", device_id=device)
        except TypeError:
            dist.init_process_group(backend=backend, init_method="env://")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return rank, local_rank, world_size, device


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def ddp_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
    args: argparse.Namespace
):
    # Set deterministic seed (offset per rank for data loader diversity, but exact model init)
    set_seed(args.seed + rank)

    is_main_process = (rank == 0)

    # 1. Verification of Multi-GPU Environment
    if is_main_process:
        print("=" * 60)
        print("KAGGLE MULTI-GPU (2x T4 DDP) CANONICAL BASELINE INITIALIZATION")
        print(f"  Visible CUDA Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  CUDA [{i}]: {props.name} | VRAM: {props.total_memory / (1024**3):.2f} GB")
        print(f"  DDP World Size:            {world_size} GPU(s)")
        print(f"  Per-GPU Batch Size:        {args.per_gpu_batch_size}")
        print(f"  Effective Global Batch:    {args.per_gpu_batch_size * world_size} (Matches audited baseline: 4)")
        print(f"  Epochs:                    {args.epochs}")
        print(f"  Learning Rate:             {args.lr}")
        print(f"  Seed:                      {args.seed}")
        print("=" * 60)

    detected_dataset = auto_detect_dataset_root(args.dataset_root)

    # 2. Load Persisted Split
    if not os.path.isfile(args.split_json):
        raise FileNotFoundError(f"Split file not found: {args.split_json}")
    with open(args.split_json, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    train_files = split_data["train_files"]
    val_files = split_data["val_files"]

    train_dataset = LowLightDataset(detected_dataset, train_files)
    val_dataset = LowLightDataset(detected_dataset, val_files)

    # 3. Distributed Sampler for Training
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_gpu_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=2 if sys.platform != "win32" else 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    # Validation DataLoader on Main Process (evaluates all 50 images at native 600x400)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    if is_main_process:
        print(f"[Dataset] Train samples: {len(train_files)} ({len(train_loader)} batches/GPU/epoch)")
        print(f"[Dataset] Validation samples: {len(val_files)} images (evaluated on Rank 0)")

    # 4. Model, DDP Wrapping, Optimizer, Loss
    model = PixelWarriorNetwork().to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Setup output dirs on Rank 0
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_desc = args.model_name if args.model_name else (
            "PixelWarriorNetwork (InstanceNorm2d)" if any(isinstance(m, nn.InstanceNorm2d) for m in model.modules())
            else "PixelWarriorNetwork (Current UNet baseline)"
        )
        config_record = {
            "experiment_id": args.experiment_id,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_desc,
            "params": sum(p.numel() for p in (model.module if world_size > 1 else model).parameters()),
            "dataset_root": detected_dataset,
            "train_samples": len(train_files),
            "val_samples": len(val_files),
            "seed": args.seed,
            "epochs": args.epochs,
            "world_size": world_size,
            "per_gpu_batch_size": args.per_gpu_batch_size,
            "effective_global_batch_size": args.per_gpu_batch_size * world_size,
            "lr": args.lr,
            "optimizer": "Adam (default betas)",
            "scheduler": "CosineAnnealingLR (eta_min=1e-6)",
            "loss": "MSELoss",
            "device": str(device),
            "gpu_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["CPU"],
            "mixed_precision": use_amp,
            "evaluation_resolution": "Native 600x400 (unwarped)",
            "clamping": "[0.0, 1.0]"
        }
        with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_record, f, indent=2)

        history = {"config": config_record, "epochs": []}

    best_val_psnr = -float("inf")
    total_start = time.time()

    if is_main_process:
        print(f"\n[Training] Commencing canonical baseline across {world_size} GPU(s)...")

    # Epoch-level progress bar (displayed only on Rank 0)
    epoch_pbar = tqdm(
        range(1, args.epochs + 1),
        desc="Total Progress",
        disable=not is_main_process,
        unit="epoch",
        dynamic_ncols=True,
        file=sys.stdout
    )

    for epoch in epoch_pbar:
        t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        running_psnr = 0.0
        num_batches = len(train_loader)

        # Batch-level progress bar (displayed only on Rank 0)
        batch_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            disable=not is_main_process,
            dynamic_ncols=True,
            leave=False,
            file=sys.stdout
        )

        for step, (low_img, high_img, _) in enumerate(batch_pbar):
            low_img = low_img.to(device)
            high_img = high_img.to(device)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(low_img)
                    loss = criterion(outputs, high_img)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(low_img)
                loss = criterion(outputs, high_img)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            with torch.no_grad():
                out_c = torch.clamp(outputs, 0.0, 1.0)
                running_psnr += calculate_psnr(out_c, high_img)

            # Update batch progress bar postfix with loss and learning rate dynamically
            if is_main_process:
                batch_pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })

        batch_pbar.close()

        # Average training loss across DDP ranks
        avg_loss_tensor = torch.tensor([running_loss / num_batches, running_psnr / num_batches], device=device)
        if world_size > 1:
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss_tensor /= world_size

        epoch_train_loss = avg_loss_tensor[0].item()
        epoch_train_psnr = avg_loss_tensor[1].item()

        # Step LR scheduler
        scheduler.step()

        # Validation on Rank 0
        if is_main_process:
            model.eval()
            val_loss_sum = 0.0
            val_psnr_list = []
            val_ssim_list = []

            # Use underlying module for clean evaluation
            eval_model = model.module if world_size > 1 else model

            with torch.no_grad():
                for low_img, high_img, _ in val_loader:
                    low_img = low_img.to(device)
                    high_img = high_img.to(device)

                    pred = eval_model(low_img)
                    pred_c = torch.clamp(pred, 0.0, 1.0)

                    val_loss_sum += criterion(pred_c, high_img).item()
                    val_psnr_list.append(calculate_psnr(pred_c, high_img))
                    val_ssim_list.append(calculate_ssim(pred_c, high_img))

            val_mse = val_loss_sum / len(val_loader)
            val_psnr = sum(val_psnr_list) / len(val_psnr_list)
            val_ssim = sum(val_ssim_list) / len(val_ssim_list)
            min_psnr = min(val_psnr_list)
            max_psnr = max(val_psnr_list)
            duration = time.time() - t0

            # Query GPU VRAM on visible devices
            vram_info = []
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**2)
                reserved = torch.cuda.memory_reserved(i) / (1024**2)
                vram_info.append(f"GPU {i}: {allocated:.0f}MB alloc / {reserved:.0f}MB res")

            # Track and save best checkpoint
            is_new_best = val_psnr > best_val_psnr
            if is_new_best:
                best_val_psnr = val_psnr
                best_path = os.path.join(checkpoint_dir, "best.pth")
                torch.save(eval_model.state_dict(), best_path)
                torch.save(eval_model.state_dict(), os.path.join(checkpoint_dir, "baseline_best.pth"))

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(epoch_train_loss, 5),
                "train_psnr": round(epoch_train_psnr, 2),
                "val_mse": round(val_mse, 6),
                "val_psnr": round(val_psnr, 2),
                "val_ssim": round(val_ssim, 4),
                "val_psnr_min": round(min_psnr, 2),
                "val_psnr_max": round(max_psnr, 2),
                "duration_sec": round(duration, 1),
                "lr": round(optimizer.param_groups[0]["lr"], 8),
                "vram_usage": vram_info
            }
            history["epochs"].append(epoch_record)

            summary_line = (
                f"\n[Epoch {epoch}/{args.epochs} Summary] "
                f"Duration: {duration:.1f}s | "
                f"Train Loss: {epoch_train_loss:.5f} | Train PSNR: {epoch_train_psnr:.2f} dB | "
                f"Val MSE: {val_mse:.6f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"VRAM: {'; '.join(vram_info)}"
            )
            tqdm.write(summary_line)

            # Update overall epoch progress bar postfix
            epoch_pbar.set_postfix({"val_psnr": f"{val_psnr:.2f}dB", "best": f"{best_val_psnr:.2f}dB"})

            # Save latest checkpoint
            torch.save(eval_model.state_dict(), os.path.join(checkpoint_dir, "latest.pth"))
            torch.save(eval_model.state_dict(), os.path.join(checkpoint_dir, "baseline_latest.pth"))
            if is_new_best:
                tqdm.write(f"  [*] Best checkpoint saved: {best_path} (PSNR: {val_psnr:.2f} dB)")

        if world_size > 1:
            dist.barrier()

    if is_main_process:
        epoch_pbar.close()

    if is_main_process:
        total_time = time.time() - total_start
        history["total_time_sec"] = round(total_time, 1)
        history["total_time_min"] = round(total_time / 60, 2)
        history["best_val_psnr"] = round(best_val_psnr, 2)

        history_path = os.path.join(args.output_dir, "train_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Generate and save convergence plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ep_data = history.get("epochs", [])
            if ep_data:
                eps = [e["epoch"] for e in ep_data]
                tr_l = [e["train_loss"] for e in ep_data]
                v_mse = [e["val_mse"] for e in ep_data]
                tr_p = [e["train_psnr"] for e in ep_data]
                v_p = [e["val_psnr"] for e in ep_data]
                v_s = [e["val_ssim"] for e in ep_data]
                lrs = [e["lr"] for e in ep_data]

                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f"{args.experiment_id} - Convergence Metrics", fontsize=14, fontweight="bold")

                axes[0, 0].plot(eps, tr_l, 'b-o', label='Train MSE Loss')
                axes[0, 0].plot(eps, v_mse, 'r--s', label='Val MSE')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('MSE Loss')
                axes[0, 0].set_title('Training vs Validation Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, linestyle='--', alpha=0.5)

                axes[0, 1].plot(eps, tr_p, 'b-o', label='Train PSNR (dB)')
                axes[0, 1].plot(eps, v_p, 'g--^', label='Val PSNR (dB)')
                axes[0, 1].axhline(best_val_psnr, color='gray', linestyle=':', label=f'Best Val: {best_val_psnr:.2f} dB')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('PSNR (dB)')
                axes[0, 1].set_title('Training vs Validation PSNR')
                axes[0, 1].legend()
                axes[0, 1].grid(True, linestyle='--', alpha=0.5)

                axes[1, 0].plot(eps, v_s, 'm-d', label='Val SSIM')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('SSIM')
                axes[1, 0].set_title('Validation SSIM')
                axes[1, 0].legend()
                axes[1, 0].grid(True, linestyle='--', alpha=0.5)

                axes[1, 1].plot(eps, lrs, 'k-o', label='Learning Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('LR')
                axes[1, 1].set_yscale('log')
                axes[1, 1].set_title('Cosine Annealing Learning Rate')
                axes[1, 1].legend()
                axes[1, 1].grid(True, linestyle='--', alpha=0.5)

                plt.tight_layout()
                plot_file = os.path.join(args.output_dir, "convergence_plot.png")
                plt.savefig(plot_file, dpi=150)
                plt.close(fig)
                print(f"  Convergence Plot:    {plot_file}")
        except Exception as e:
            print(f"  [Warning] Could not generate convergence plot: {e}")

        print("\n" + "=" * 60)
        print("2x T4 DDP TRAINING COMPLETE")
        print(f"  Experiment ID:       {args.experiment_id}")
        print(f"  Total Duration:      {total_time/60:.2f} minutes")
        print(f"  Average Epoch Time:  {total_time/args.epochs:.1f} seconds/epoch")
        print(f"  Best Val PSNR:       {best_val_psnr:.2f} dB")
        print(f"  Saved Artifacts:     {args.output_dir}")
        print("=" * 60)

    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="Kaggle 2x T4 DDP Training Pipeline")
    parser.add_argument("--dataset_root", type=str, default=None, help="Dataset root dir")
    parser.add_argument("--split_json", "--split_file", dest="split_json", type=str, default="splits/split_seed42.json", help="Split JSON path")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/baseline", help="Outputs dir")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--per_gpu_batch_size", type=int, default=2, help="Per-GPU batch size (2 GPUs * 2 = effective batch size 4)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment_id", type=str, default="EXP-01-KAGGLE-2xT4-DDP-BASELINE", help="Experiment identifier")
    parser.add_argument("--model_name", type=str, default=None, help="Model name description")
    args = parser.parse_args()

    rank, local_rank, world_size, device = setup_ddp()
    ddp_worker(rank, local_rank, world_size, device, args)


if __name__ == "__main__":
    main()
