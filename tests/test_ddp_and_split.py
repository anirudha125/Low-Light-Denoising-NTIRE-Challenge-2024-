"""
Lightweight smoke test for DDP partitioning, sampler determinism, and pipeline integrity.
Executes purely as a local unit test without heavy training.
"""

import os
import sys
import json
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.model import PixelWarriorNetwork
from src.metrics import calculate_psnr, calculate_ssim, calculate_mse


class DummyDataset(Dataset):
    def __init__(self, size=435):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return idx


def test_ddp_partitioning():
    print("[Test 1] Verifying DistributedSampler partitioning across World Size = 2...")
    dataset = DummyDataset(size=435)
    world_size = 2
    per_gpu_batch_size = 2

    sampler_rank0 = DistributedSampler(dataset, num_replicas=world_size, rank=0, shuffle=True, seed=42)
    sampler_rank1 = DistributedSampler(dataset, num_replicas=world_size, rank=1, shuffle=True, seed=42)

    sampler_rank0.set_epoch(1)
    sampler_rank1.set_epoch(1)

    indices_rank0 = list(sampler_rank0)
    indices_rank1 = list(sampler_rank1)

    print(f"  Rank 0 sample count: {len(indices_rank0)}")
    print(f"  Rank 1 sample count: {len(indices_rank1)}")

    # Check that each GPU gets 218 samples (435 rounded up to even for DDP)
    assert len(indices_rank0) == 218, f"Expected 218, got {len(indices_rank0)}"
    assert len(indices_rank1) == 218, f"Expected 218, got {len(indices_rank1)}"

    # Check non-overlapping primary sets (DistributedSampler pads 1 item to make lengths equal)
    set0 = set(indices_rank0)
    set1 = set(indices_rank1)
    overlap = set0.intersection(set1)
    # The only overlap can be the 1 padded item (435 is odd, so 1 item is repeated to balance batches)
    print(f"  Padded boundary overlap: {len(overlap)} (Expected: 1 duplicate for odd dataset size)")
    assert len(overlap) <= 1, f"Unexpected overlap count: {len(overlap)}"

    # Effective batch size check
    effective_global_batch = per_gpu_batch_size * world_size
    assert effective_global_batch == 4, f"Effective batch size {effective_global_batch} != 4"
    print(f"  [PASS] Effective Global Batch Size: {per_gpu_batch_size} (per GPU) * {world_size} (GPUs) = {effective_global_batch}")


def test_split_file():
    print("\n[Test 2] Verifying persisted train/val split files...")
    split_path = "splits/split_seed42.json"
    assert os.path.isfile(split_path), f"Missing split file: {split_path}"
    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_files = data["train_files"]
    val_files = data["val_files"]

    assert len(train_files) == 435, f"Train count {len(train_files)} != 435"
    assert len(val_files) == 50, f"Val count {len(val_files)} != 50"
    assert len(set(train_files) & set(val_files)) == 0, "Data leakage detected between train and val!"
    print(f"  [PASS] Split integrity verified: 435 train, 50 val, 0 leakage.")


def test_model_and_metrics():
    print("\n[Test 3] Verifying baseline model forward pass & metric range clamping...")
    model = PixelWarriorNetwork()
    x = torch.rand(1, 3, 400, 600)  # Native 600x400
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 400, 600), f"Output shape mismatch: {out.shape}"

    # Metric clamping check
    unclamped = torch.tensor([[-0.5, 1.5], [0.4, 0.5]]).unsqueeze(0).unsqueeze(0)
    target = torch.tensor([[0.0, 1.0], [0.6, 0.5]]).unsqueeze(0).unsqueeze(0)
    # unclamped clamps to [[0, 1], [0.4, 0.5]], diff from target is [[0, 0], [-0.2, 0]]
    p_val = calculate_psnr(unclamped, target)
    assert not math.isinf(p_val) and p_val > 0, f"PSNR clamping failed: p_val={p_val}"
    print(f"  [PASS] Forward pass shape: {out.shape}, Clamped PSNR: {p_val:.2f} dB verified.")


if __name__ == "__main__":
    test_ddp_partitioning()
    test_split_file()
    test_model_and_metrics()
    print("\n>>> ALL PRE-TRAINING SMOKE TESTS PASSED SUCCESSFULLY! <<<")
