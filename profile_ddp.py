"""
profile_ddp.py
--------------
Standalone DeepSpeed DDP profiler for SoundStream.

Goals:
  1. Time each phase of the step on every rank separately
       (data wait, forward, backward, all_reduce, optimizer)
  2. Detect per-rank imbalance (NUMA / slow-link rank)
  3. Enable NCCL_DEBUG=INFO + NCCL profiling for fine-grained comm stats

Launch (same as main_ddp.py):
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL \
    deepspeed --num_gpus=6 profile_ddp.py \
              --deepspeed_config ds_config.json

Outputs:
    - profile_ddp_rank<R>.log   per-rank CSV-style log
    - profile_ddp_summary.txt   aggregated by rank 0
"""
import argparse
import os
import time
import json
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import deepspeed
from net import SoundStream, WaveDiscriminator, STFTDiscriminator
from dataset import NSynthDataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
parser.add_argument("--profile_batches", type=int, default=30,
                    help="how many batches to profile per rank")
parser.add_argument("--warmup_batches", type=int, default=5,
                    help="warmup before timing")
parser.add_argument("--log_dir", type=str, default="profile_logs")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Init distributed
# ---------------------------------------------------------------------------
deepspeed.init_distributed()
local_rank = int(os.environ["LOCAL_RANK"])
rank       = dist.get_rank()
world_size = dist.get_world_size()
device     = torch.device(f"cuda:{local_rank}")

os.makedirs(args.log_dir, exist_ok=True)
LOG_PATH = os.path.join(args.log_dir, f"profile_ddp_rank{rank}.log")


def log(msg):
    """Append a line to per-rank log file (opened in append, flushed)."""
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")
    if rank == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Build a tiny JointModel + DeepSpeed engine
# ---------------------------------------------------------------------------
class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.soundstream = SoundStream(C=32, D=128, n_q=8, codebook_size=1024)
        self.wave_disc   = WaveDiscriminator(num_D=3, downsampling_factor=2)
        self.stft_disc   = STFTDiscriminator(C=1, F_bins=1024 // 2)


joint_model = JointModel()
model_engine, optimizer, _, scheduler = deepspeed.initialize(
    args=args, model=joint_model,
    model_parameters=joint_model.parameters(),
)
soundstream = model_engine.module.soundstream
wave_disc   = model_engine.module.wave_disc
stft_disc   = model_engine.module.stft_disc


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def collate_fn(batch):
    lengths = torch.tensor([elem.shape[-1] for elem in batch])
    return nn.utils.rnn.pad_sequence(batch, batch_first=True), lengths


train_dataset = NSynthDataset(
    audio_dir="/M101/dataset/audio/NSynth/nsynth-train/audio")
train_sampler = DistributedSampler(
    train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
train_loader = DataLoader(
    train_dataset, batch_size=16, sampler=train_sampler,
    collate_fn=collate_fn, num_workers=2)
sr = train_dataset.sr


# ---------------------------------------------------------------------------
# Loss funcs (copied verbatim from main.py / main_ddp.py — same numbers)
# ---------------------------------------------------------------------------
def adversarial_g_loss(features_stft_disc_G_x, features_wave_disc_G_x,
                       lengths_stft, lengths_wave):
    wave_disc_names = lengths_wave.keys()
    stft_loss = F.relu(1-features_stft_disc_G_x[-1]).sum(dim=3).squeeze()/lengths_stft[-1].squeeze()
    wave_loss = torch.cat([
        F.relu(1-features_wave_disc_G_x[k][-1]).sum(dim=2).squeeze()/lengths_wave[k][-1].squeeze()
        for k in wave_disc_names
    ])
    return torch.cat([stft_loss, wave_loss]).mean()


def feature_loss(features_stft_disc_x, features_wave_disc_x,
                 features_stft_disc_G_x, features_wave_disc_G_x,
                 lengths_wave, lengths_stft):
    wave_disc_names = lengths_wave.keys()
    stft_loss = torch.stack([
        ((fx-fGx).abs().sum(dim=-1)/lengths_stft[i].view(-1,1,1)).sum(dim=-1).sum(dim=-1)
        for i, (fx, fGx) in enumerate(zip(features_stft_disc_x, features_stft_disc_G_x))
    ], dim=1).mean(dim=1, keepdim=True)
    wave_loss = torch.stack([
        torch.stack([
            (fx-fGx).abs().sum(dim=-1).sum(dim=-1)/lengths_wave[k][i]
            for i, (fx, fGx) in enumerate(zip(features_wave_disc_x[k], features_wave_disc_G_x[k]))
        ], dim=1) for k in wave_disc_names
    ], dim=2).mean(dim=1)
    return torch.cat([stft_loss, wave_loss], dim=1).mean()


def spectral_reconstruction_loss(x, G_x, eps=1e-4):
    L = 0
    for i in range(6, 12):
        s = 2**i
        alpha_s = (s/2)**0.5
        melspec = torch.nn.Sequential().to(device)  # placeholder; we'll redo below
        # Build melspec on the fly (matches main_ddp.py behavior).
        from torchaudio.transforms import MelSpectrogram
        ms = MelSpectrogram(sample_rate=sr, n_fft=s, hop_length=s//4,
                            n_mels=8,
                            wkwargs={"device": device}).to(device)
        S_x   = ms(x)
        S_G_x = ms(G_x)
        loss = (S_x-S_G_x).abs().sum() + alpha_s * (
            ((torch.log(S_x.abs()+eps)-torch.log(S_G_x.abs()+eps))**2)
            .sum(dim=-2)**0.5).sum()
        L += loss
    return L


def adversarial_d_loss(features_stft_disc_x, features_wave_disc_x,
                       features_stft_disc_G_x, features_wave_disc_G_x,
                       lengths_stft, lengths_wave):
    wave_disc_names = lengths_wave.keys()
    real_stft_loss = F.relu(1-features_stft_disc_x[-1]).sum(dim=3).squeeze()/lengths_stft[-1].squeeze()
    real_wave_loss = torch.stack([
        F.relu(1-features_wave_disc_x[k][-1]).sum(dim=-1).squeeze()/lengths_wave[k][-1].squeeze()
        for k in wave_disc_names], dim=1)
    real_loss = torch.cat([real_stft_loss.view(-1,1), real_wave_loss], dim=1).mean()
    gen_stft_loss = F.relu(1+features_stft_disc_G_x[-1]).sum(dim=-1).squeeze()/lengths_stft[-1].squeeze()
    gen_wave_loss = torch.stack([
        F.relu(1+features_wave_disc_G_x[k][-1]).sum(dim=-1).squeeze()/lengths_wave[k][-1].squeeze()
        for k in wave_disc_names], dim=1)
    gen_loss = torch.cat([gen_stft_loss.view(-1,1), gen_wave_loss], dim=1).mean()
    return real_loss + gen_loss


criterion_g = lambda x, G_x, fs_x, fs_w, fs_Gx, fs_wGx, lw, lst: (
    1 * adversarial_g_loss(fs_Gx, fs_wGx, lst, lw)
  + 100 * feature_loss(fs_x, fs_w, fs_Gx, fs_wGx, lw, lst)
  + 1 * spectral_reconstruction_loss(x, G_x)
)
criterion_d = adversarial_d_loss


# ---------------------------------------------------------------------------
# Per-rank timing accumulators
# ---------------------------------------------------------------------------
# Each entry: dict of phase durations in milliseconds for one batch.
samples = []


def stamp(t0):
    """Return milliseconds elapsed since t0, after a host-side record."""
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Profiling loop
# ---------------------------------------------------------------------------
log(f"=== profile_ddp.py started on rank {rank}/{world_size} "
    f"(device=cuda:{local_rank}) ===")
log(f"NCCL version: {torch.cuda.nccl.version()}")
log(f"GPU: {torch.cuda.get_device_name(local_rank)}")
log(f"profile_batches={args.profile_batches} warmup={args.warmup_batches}")

model_engine.module.train()

for step, (x, lengths_x) in enumerate(train_loader):
    is_warmup = step < args.warmup_batches
    do_measure = step >= args.warmup_batches and step < args.warmup_batches + args.profile_batches

    # ---------- DATA WAIT ----------
    t = time.perf_counter()
    x = x.to(device, non_blocking=True)
    lengths_x = lengths_x.to(device, non_blocking=True)
    torch.cuda.synchronize(device)
    t_data_wait = (time.perf_counter() - t) * 1000.0

    # ---------- G FORWARD ----------
    t = time.perf_counter()
    G_x = soundstream(x)
    torch.cuda.synchronize(device)
    t_g_fwd = (time.perf_counter() - t) * 1000.0

    # ---------- STFT (CPU+GPU mix) ----------
    t = time.perf_counter()
    s_x = torch.stft(x.squeeze(), n_fft=1024, hop_length=256,
                     window=torch.hann_window(window_length=1024, device=device),
                     return_complex=False).permute(0, 3, 1, 2)
    lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")
    s_G_x = torch.stft(G_x.squeeze(), n_fft=1024, hop_length=256,
                       window=torch.hann_window(window_length=1024, device=device),
                       return_complex=False).permute(0, 3, 1, 2)
    torch.cuda.synchronize(device)
    t_stft = (time.perf_counter() - t) * 1000.0

    # ---------- DISC FORWARD ----------
    t = time.perf_counter()
    lengths_stft = stft_disc.features_lengths(lengths_s_x)
    lengths_wave = wave_disc.features_lengths(lengths_x)
    f_stft_x = stft_disc(s_x)
    f_wave_x = wave_disc(x)
    f_stft_Gx = stft_disc(s_G_x)
    f_wave_Gx = wave_disc(G_x)
    torch.cuda.synchronize(device)
    t_disc_fwd = (time.perf_counter() - t) * 1000.0

    # ---------- LOSS + G BACKWARD + STEP ----------
    t = time.perf_counter()
    loss_g = criterion_g(x, G_x, f_stft_x, f_wave_x, f_stft_Gx, f_wave_Gx,
                         lengths_wave, lengths_stft)
    model_engine.zero_grad()
    model_engine.backward(loss_g)
    model_engine.step()
    torch.cuda.synchronize(device)
    t_g_step = (time.perf_counter() - t) * 1000.0

    # ---------- D BACKWARD + STEP ----------
    t = time.perf_counter()
    f_stft_x2 = stft_disc(s_x)
    f_wave_x2 = wave_disc(x)
    f_stft_Gx_det = stft_disc(s_G_x.detach())
    f_wave_Gx_det = wave_disc(G_x.detach())
    loss_d = criterion_d(f_stft_x2, f_wave_x2,
                         f_stft_Gx_det, f_wave_Gx_det,
                         lengths_stft, lengths_wave)
    model_engine.zero_grad()
    model_engine.backward(loss_d)
    model_engine.step()
    torch.cuda.synchronize(device)
    t_d_step = (time.perf_counter() - t) * 1000.0

    if do_measure:
        samples.append({
            "step": step,
            "data_wait_ms": t_data_wait,
            "g_fwd_ms":     t_g_fwd,
            "stft_ms":      t_stft,
            "disc_fwd_ms":  t_disc_fwd,
            "g_step_ms":    t_g_step,
            "d_step_ms":    t_d_step,
            "total_ms":     t_data_wait + t_g_fwd + t_stft
                          + t_disc_fwd + t_g_step + t_d_step,
        })

    if step % 5 == 0:
        log(f"[rank{rank}] step={step} total={t_data_wait + t_g_fwd + t_stft + t_disc_fwd + t_g_step + t_d_step:.1f}ms "
            f"(data={t_data_wait:.1f} g_fwd={t_g_fwd:.1f} stft={t_stft:.1f} "
            f"disc={t_disc_fwd:.1f} g_step={t_g_step:.1f} d_step={t_d_step:.1f})")

    if not is_warmup and step >= args.warmup_batches + args.profile_batches:
        break

log(f"[rank{rank}] profiling done. {len(samples)} samples collected.")

# ---------------------------------------------------------------------------
# Aggregate per-rank stats and gather to rank 0
# ---------------------------------------------------------------------------
def stats(values):
    if not values:
        return (0.0, 0.0, 0.0)
    return (statistics.mean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0,
            max(values))


local_summary = {
    "rank": rank,
    "n_samples": len(samples),
}
for key in ["total_ms", "data_wait_ms", "g_fwd_ms", "stft_ms",
            "disc_fwd_ms", "g_step_ms", "d_step_ms"]:
    m, s, mx = stats([s_[key] for s_ in samples])
    local_summary[f"{key}_mean"] = m
    local_summary[f"{key}_std"]  = s
    local_summary[f"{key}_max"]  = mx

# Gather to rank 0
# NOTE: gather_object requires that non-dst ranks pass gather_list=None,
# otherwise PyTorch raises "Argument gather_list must NOT be specified on
# non-destination ranks".  See torch/distributed/distributed_c10d.py:3029.
if rank == 0:
    gathered = [None] * world_size
    dist.gather_object(local_summary, gathered, dst=0)
else:
    dist.gather_object(local_summary, None, dst=0)

# Also dump the per-step trace this rank collected so we can read it even
# if the gather fails (rank 0 only — others already wrote their own files).
if rank == 0:
    rank0_path = os.path.join(args.log_dir, "profile_ddp_rank0_trace.txt")
    with open(rank0_path, "w") as f:
        f.write("# step  data  g_fwd  stft  disc  g_step  d_step  total\n")
        for s in samples:
            f.write(f"{s['step']:>4} "
                    f"{s['data_wait_ms']:>6.2f} {s['g_fwd_ms']:>6.2f} "
                    f"{s['stft_ms']:>6.2f} {s['disc_fwd_ms']:>6.2f} "
                    f"{s['g_step_ms']:>7.2f} {s['d_step_ms']:>7.2f} "
                    f"{s['total_ms']:>7.2f}\n")

if rank == 0:
    out = []
    for s in gathered:
        if s is None:
            continue
        out.append(s)

    summary_path = os.path.join(args.log_dir, "profile_ddp_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"world_size = {world_size}\n")
        f.write(f"profile_batches = {args.profile_batches}\n\n")
        f.write(f"{'rank':>4} {'n':>4} "
                f"{'total_mean':>11} {'total_std':>10} {'total_max':>10} "
                f"{'data_mean':>10} {'g_fwd':>8} {'stft':>8} "
                f"{'disc_fwd':>10} {'g_step':>10} {'d_step':>10}\n")
        for s in sorted(out, key=lambda x: x["rank"]):
            f.write(
                f"{s['rank']:>4} {s['n_samples']:>4} "
                f"{s['total_ms_mean']:>11.2f} {s['total_ms_std']:>10.2f} "
                f"{s['total_ms_max']:>10.2f} "
                f"{s['data_wait_ms_mean']:>10.2f} {s['g_fwd_ms_mean']:>8.2f} "
                f"{s['stft_ms_mean']:>8.2f} {s['disc_fwd_ms_mean']:>10.2f} "
                f"{s['g_step_ms_mean']:>10.2f} {s['d_step_ms_mean']:>10.2f}\n"
            )

        # Slowest rank identification
        slowest = max(out, key=lambda x: x["total_ms_mean"])
        fastest = min(out, key=lambda x: x["total_ms_mean"])
        f.write(f"\nslowest rank = {slowest['rank']}  "
                f"({slowest['total_ms_mean']:.2f} ms/batch)\n")
        f.write(f"fastest rank = {fastest['rank']}  "
                f"({fastest['total_ms_mean']:.2f} ms/batch)\n")
        f.write(f"imbalance    = "
                f"{(slowest['total_ms_mean']/fastest['total_ms_mean']-1)*100:.1f}%\n")

        # Per-phase breakdown averaged across ranks
        f.write("\n--- avg across all ranks ---\n")
        for phase in ["data_wait_ms", "g_fwd_ms", "stft_ms",
                      "disc_fwd_ms", "g_step_ms", "d_step_ms"]:
            vals = [s_[f"{phase}_mean"] for s_ in out]
            f.write(f"{phase:>16}: mean={statistics.mean(vals):.2f}ms "
                    f"std={statistics.stdev(vals):.2f}ms "
                    f"max={max(vals):.2f}ms "
                    f"min={min(vals):.2f}ms\n")
    print(f"[rank0] summary written to {summary_path}", flush=True)
    with open(summary_path) as f:
        print(f.read())

dist.barrier()