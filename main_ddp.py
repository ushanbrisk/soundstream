"""
DeepSpeed DDP data-parallel training script for SoundStream.

Based on main.py. Differences vs. main.py are summarized in tutorial/ddp.md.

Launch:
    deepspeed --num_gpus=6 main_ddp.py --deepspeed_config ds_config.json

Optional flags:
    --resume                 : resume from checkpoints/latest
    --ckpt_dir PATH          : where to save (default: ./checkpoints)
    --local_rank INT         : auto-set by deepspeed launcher
"""

import argparse
import json
import os

import deepspeed
import swanlab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

from net import SoundStream, WaveDiscriminator, STFTDiscriminator
from dataset import NSynthDataset


# ----------------------------------------------------------------------------
# 1. CLI args.  --local_rank is required so DeepSpeed's launcher can inject it.
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1,
                    help="local rank passed by `deepspeed` launcher")
parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
parser.add_argument("--resume", action="store_true",
                    help="resume training from checkpoints/latest")
parser.add_argument("--swanlab_project", type=str, default="soundstream")
parser.add_argument("--swanlab_workspace", type=str, default=None,
                    help="SwanLab workspace (equivalent of wandb 'entity')")
parser.add_argument("--swanlab_run_name", type=str, default=None)
parser.add_argument("--swanlab_mode", type=str, default="cloud",
                    choices=["cloud", "cloud-only", "local", "disabled"],
                    help="SwanLab run mode. 'cloud' needs SWANLAB_API_KEY; "
                         "'local' stores only in ./swanlog")
args = parser.parse_args()


# ----------------------------------------------------------------------------
# 2. Distributed init.  Must come BEFORE any per-rank setup that touches CUDA.
# ----------------------------------------------------------------------------
deepspeed.init_distributed()
local_rank = int(os.environ["LOCAL_RANK"])
rank       = dist.get_rank()
world_size = dist.get_world_size()
device     = torch.device(f"cuda:{local_rank}")

# IMPORTANT: do NOT add an "is main process" early-return here.  DeepSpeed's
# launcher runs one Python process per GPU, and every one of them must
# participate in the distributed group, otherwise all_reduce blocks forever.


# ----------------------------------------------------------------------------
# 3. Hyperparameters (kept identical to main.py).
# ----------------------------------------------------------------------------
LAMBDA_ADV = 1
LAMBDA_FEAT = 100
LAMBDA_REC = 1
N_EPOCHS = 2
BATCH_SIZE = 16


# ----------------------------------------------------------------------------
# 4. JointModel container.
#
# DeepSpeed's `initialize()` takes ONE nn.Module.  We wrap the generator and
# the two discriminators together so a single optimizer handles them all.
# Because `loss_g` only carries gradients to the generator side and `loss_d`
# only carries gradients to the discriminator side (the other side is detached),
# Adam.step() naturally only updates the parameters that have non-None grads.
# ----------------------------------------------------------------------------
class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.soundstream = SoundStream(C=32, D=128, n_q=8, codebook_size=1024)
        self.wave_disc   = WaveDiscriminator(num_D=3, downsampling_factor=2)
        self.stft_disc   = STFTDiscriminator(C=1, F_bins=1024 // 2)


joint_model = JointModel()

# Let deepspeed.initialize() load `ds_config.json` itself (driven by the
# --deepspeed_config flag we registered in argparse).  Passing a `config=` dict
# here AND having the flag set raises:
#     AssertionError: Not sure how to proceed, we were given deepspeed configs
#     in the deepspeed arguments and deepspeed.initialize() function call
# So we choose the argparse path: DeepSpeed reads + validates the JSON.
model_engine, optimizer, _, scheduler = deepspeed.initialize(
    args=args,
    model=joint_model,
    model_parameters=joint_model.parameters(),
)

# Convenience references.  Always go through `model_engine.module.X` so that
# the same code works under ZeRO (where the parameters may live on different
# ranks) or under stage 0 (where the module is the plain underlying model).
soundstream = model_engine.module.soundstream
wave_disc   = model_engine.module.wave_disc
stft_disc   = model_engine.module.stft_disc


# ----------------------------------------------------------------------------
# 5. Data: each rank gets its own shard via DistributedSampler.
# ----------------------------------------------------------------------------
def collate_fn(batch):
    lengths = torch.tensor([elem.shape[-1] for elem in batch])
    return nn.utils.rnn.pad_sequence(batch, batch_first=True), lengths


train_dataset = NSynthDataset(audio_dir="/M101/dataset/audio/NSynth/nsynth-train/audio")
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                           collate_fn=collate_fn, num_workers=2)
sr = train_dataset.sr

valid_dataset = NSynthDataset(audio_dir="/M101/dataset/audio/NSynth/nsynth-valid/audio")
valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
valid_loader  = DataLoader(valid_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler,
                           collate_fn=collate_fn, num_workers=2)

test_dataset = NSynthDataset(audio_dir="/M101/dataset/audio/NSynth/nsynth-test/audio")
test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler,
                          collate_fn=collate_fn, num_workers=2)


# ----------------------------------------------------------------------------
# 6. Loss functions — copy-paste from main.py, no changes.
# ----------------------------------------------------------------------------
def adversarial_g_loss(features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave):
    wave_disc_names = lengths_wave.keys()

    stft_loss = F.relu(1-features_stft_disc_G_x[-1]).sum(dim=3).squeeze()/lengths_stft[-1].squeeze()
    wave_loss = torch.cat([F.relu(1-features_wave_disc_G_x[key][-1]).sum(dim=2).squeeze()/lengths_wave[key][-1].squeeze() for key in wave_disc_names])
    loss = torch.cat([stft_loss, wave_loss]).mean()

    return loss


def feature_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft):
    wave_disc_names = lengths_wave.keys()

    stft_loss = torch.stack([((feat_x-feat_G_x).abs().sum(dim=-1)/lengths_stft[i].view(-1,1,1)).sum(dim=-1).sum(dim=-1) for i, (feat_x, feat_G_x) in enumerate(zip(features_stft_disc_x, features_stft_disc_G_x))], dim=1).mean(dim=1, keepdim=True)
    wave_loss = torch.stack([torch.stack([(feat_x-feat_G_x).abs().sum(dim=-1).sum(dim=-1)/lengths_wave[key][i] for i, (feat_x, feat_G_x) in enumerate(zip(features_wave_disc_x[key], features_wave_disc_G_x[key]))], dim=1) for key in wave_disc_names], dim=2).mean(dim=1)
    loss = torch.cat([stft_loss, wave_loss], dim=1).mean()

    return loss


def spectral_reconstruction_loss(x, G_x, eps=1e-4):
    L = 0
    for i in range(6,12):
        s = 2**i
        alpha_s = (s/2)**0.5
        melspec = MelSpectrogram(sample_rate=sr, n_fft=s, hop_length=s//4, n_mels=8, wkwargs={"device": device}).to(device)
        S_x = melspec(x)
        S_G_x = melspec(G_x)

        loss = (S_x-S_G_x).abs().sum() + alpha_s*(((torch.log(S_x.abs()+eps)-torch.log(S_G_x.abs()+eps))**2).sum(dim=-2)**0.5).sum()
        L += loss

    return L


def adversarial_d_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave):
    wave_disc_names = lengths_wave.keys()

    real_stft_loss = F.relu(1-features_stft_disc_x[-1]).sum(dim=3).squeeze()/lengths_stft[-1].squeeze()
    real_wave_loss = torch.stack([F.relu(1-features_wave_disc_x[key][-1]).sum(dim=-1).squeeze()/lengths_wave[key][-1].squeeze() for key in wave_disc_names], dim=1)
    real_loss = torch.cat([real_stft_loss.view(-1,1), real_wave_loss], dim=1).mean()

    generated_stft_loss = F.relu(1+features_stft_disc_G_x[-1]).sum(dim=-1).squeeze()/lengths_stft[-1].squeeze()
    generated_wave_loss = torch.stack([F.relu(1+features_wave_disc_G_x[key][-1]).sum(dim=-1).squeeze()/lengths_wave[key][-1].squeeze() for key in wave_disc_names], dim=1)
    generated_loss = torch.cat([generated_stft_loss.view(-1,1), generated_wave_loss], dim=1).mean()

    return real_loss + generated_loss


criterion_g = lambda x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft: LAMBDA_ADV*adversarial_g_loss(features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave) + LAMBDA_FEAT*feature_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft) + LAMBDA_REC*spectral_reconstruction_loss(x.float(), G_x.float())
criterion_d = adversarial_d_loss


# ----------------------------------------------------------------------------
# 7. Checkpoint directory — created on rank 0, others wait at the barrier.
# ----------------------------------------------------------------------------
if rank == 0:
    os.makedirs(args.ckpt_dir, exist_ok=True)
dist.barrier()


# ----------------------------------------------------------------------------
# 8. Resume (optional) — load DeepSpeed checkpoint on every rank.
# ----------------------------------------------------------------------------
start_epoch   = 1
best_val_loss = float("inf")
if args.resume:
    # load_checkpoint is collective; every rank must call it.
    _, client_state = model_engine.load_checkpoint(args.ckpt_dir, "latest")
    if client_state is not None:
        start_epoch   = client_state.get("epoch", 0) + 1
        best_val_loss = client_state.get("best_val_loss", float("inf"))
    if rank == 0:
        print(f"[resume] starting from epoch {start_epoch}, best_val_loss={best_val_loss}")


# ----------------------------------------------------------------------------
# 9. Per-epoch history dict + SwanLab init — only meaningful on rank 0.
# ----------------------------------------------------------------------------
if rank == 0:
    history = {
        "train": {"d": [], "g": []},
        "valid": {"d": [], "g": []},
        "test":  {"d": [], "g": []},
    }
    global_step = 0
    swanlab.init(
        project=args.swanlab_project,
        workspace=args.swanlab_workspace,
        experiment_name=args.swanlab_run_name,
        config={
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "lambda_adv": LAMBDA_ADV,
            "lambda_feat": LAMBDA_FEAT,
            "lambda_rec": LAMBDA_REC,
            "lr": 1e-4,
            "betas": [0.5, 0.9],
            "gradient_clipping": 1.0,
            "world_size": world_size,
            "model_C": 32, "model_D": 128,
            "model_n_q": 8, "model_codebook_size": 1024,
            "wave_disc_num_D": 3, "wave_disc_downsampling_factor": 2,
            "stft_disc_C": 1, "stft_disc_F_bins": 1024 // 2,
            "bf16_enabled": False,
            "zero_stage": 0,
            "deepspeed_version": deepspeed.__version__,
            "torch_version": torch.__version__,
        },
        mode=args.swanlab_mode,
        logdir=os.path.join(args.ckpt_dir, "swanlog"),
        job_type="train",
    )
else:
    history = None
    global_step = 0


# ----------------------------------------------------------------------------
# 10. Helper: aggregate a (sum, count) scalar across ranks.
# ----------------------------------------------------------------------------
def all_reduce_mean(local_sum: float, local_count: int):
    """Return (global_mean, global_count).  Works from every rank."""
    t = torch.tensor([local_sum, local_count], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t[0] / t[1]).item(), t[1].item()


# ----------------------------------------------------------------------------
# 11. Training loop.
# ----------------------------------------------------------------------------
for epoch in range(start_epoch, N_EPOCHS + 1):

    # ---- train ----
    model_engine.module.train()
    train_sampler.set_epoch(epoch)   # crucial: ensures different shuffle each epoch

    local_train_sum_g = 0.0
    local_train_sum_d = 0.0
    local_train_cnt   = 0

    for x, lengths_x in tqdm(train_loader, disable=(rank != 0), desc=f"[train e{epoch}]"):
        x = x.to(device)
        lengths_x = lengths_x.to(device)

        G_x = soundstream(x)

        s_x = torch.stft(x.squeeze(), n_fft=1024, hop_length=256,
                         window=torch.hann_window(window_length=1024, device=device),
                         return_complex=False).permute(0, 3, 1, 2)
        lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")
        s_G_x = torch.stft(G_x.squeeze(), n_fft=1024, hop_length=256,
                           window=torch.hann_window(window_length=1024, device=device),
                           return_complex=False).permute(0, 3, 1, 2)

        lengths_stft = stft_disc.features_lengths(lengths_s_x)
        lengths_wave = wave_disc.features_lengths(lengths_x)

        features_stft_disc_x   = stft_disc(s_x)
        features_wave_disc_x   = wave_disc(x)
        features_stft_disc_G_x = stft_disc(s_G_x)
        features_wave_disc_G_x = wave_disc(G_x)

        # ---- generator step ----
        loss_g = criterion_g(x, G_x,
                             features_stft_disc_x, features_wave_disc_x,
                             features_stft_disc_G_x, features_wave_disc_G_x,
                             lengths_wave, lengths_stft)
        model_engine.zero_grad()
        model_engine.backward(loss_g)
        model_engine.step()

        # ---- discriminator step ----
        # Recompute features; generated ones are detached so gradients don't
        # leak back into the generator side.
        features_stft_disc_x      = stft_disc(s_x)
        features_wave_disc_x      = wave_disc(x)
        features_stft_disc_G_x_det = stft_disc(s_G_x.detach())
        features_wave_disc_G_x_det = wave_disc(G_x.detach())

        loss_d = criterion_d(features_stft_disc_x, features_wave_disc_x,
                             features_stft_disc_G_x_det, features_wave_disc_G_x_det,
                             lengths_stft, lengths_wave)
        model_engine.zero_grad()
        model_engine.backward(loss_d)
        model_engine.step()

        local_train_sum_g += loss_g.item()
        local_train_sum_d += loss_d.item()
        local_train_cnt   += 1

        if rank == 0:
            swanlab.log({
                "train_step/g_loss": loss_g.item(),
                "train_step/d_loss": loss_d.item(),
                "epoch": epoch,
            }, step=global_step)
            global_step += 1

    # Cross-rank aggregation: each rank only saw its shard of the dataset,
    # so we average over the whole world.
    avg_train_g, train_n = all_reduce_mean(local_train_sum_g, local_train_cnt)
    avg_train_d, _       = all_reduce_mean(local_train_sum_d, local_train_cnt)

    dist.barrier()
    if rank == 0:
        history["train"]["g"].append(avg_train_g)
        history["train"]["d"].append(avg_train_d)
        swanlab.log({
            "epoch": epoch,
            "train/g_loss": avg_train_g,
            "train/d_loss": avg_train_d,
            "train/batches": int(train_n),
        }, step=global_step)
        print(f"[epoch {epoch}][train] g={avg_train_g:.4f}  d={avg_train_d:.4f}  "
              f"(batches: {int(train_n)})")

    # ---- validation ----
    model_engine.module.eval()
    valid_sampler.set_epoch(epoch)

    local_valid_sum_g = 0.0
    local_valid_sum_d = 0.0
    local_valid_cnt   = 0

    with torch.no_grad():
        for x, lengths_x in tqdm(valid_loader, disable=(rank != 0), desc=f"[valid e{epoch}]"):
            x = x.to(device)
            lengths_x = lengths_x.to(device)

            G_x = soundstream(x)

            s_x = torch.stft(x.squeeze(), n_fft=1024, hop_length=256,
                             window=torch.hann_window(window_length=1024, device=device),
                             return_complex=False).permute(0, 3, 1, 2)
            lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")
            s_G_x = torch.stft(G_x.squeeze(), n_fft=1024, hop_length=256,
                               window=torch.hann_window(window_length=1024, device=device),
                               return_complex=False).permute(0, 3, 1, 2)

            lengths_stft = stft_disc.features_lengths(lengths_s_x)
            lengths_wave = wave_disc.features_lengths(lengths_x)

            features_stft_disc_x   = stft_disc(s_x)
            features_wave_disc_x   = wave_disc(x)
            features_stft_disc_G_x = stft_disc(s_G_x)
            features_wave_disc_G_x = wave_disc(G_x)

            loss_g = criterion_g(x, G_x,
                                 features_stft_disc_x, features_wave_disc_x,
                                 features_stft_disc_G_x, features_wave_disc_G_x,
                                 lengths_wave, lengths_stft)

            features_stft_disc_x      = stft_disc(s_x)
            features_wave_disc_x      = wave_disc(x)
            features_stft_disc_G_x_det = stft_disc(s_G_x.detach())
            features_wave_disc_G_x_det = wave_disc(G_x.detach())

            loss_d = criterion_d(features_stft_disc_x, features_wave_disc_x,
                                 features_stft_disc_G_x_det, features_wave_disc_G_x_det,
                                 lengths_stft, lengths_wave)

            local_valid_sum_g += loss_g.item()
            local_valid_sum_d += loss_d.item()
            local_valid_cnt   += 1

    avg_valid_g, valid_n = all_reduce_mean(local_valid_sum_g, local_valid_cnt)
    avg_valid_d, _       = all_reduce_mean(local_valid_sum_d, local_valid_cnt)

    dist.barrier()
    if rank == 0:
        history["valid"]["g"].append(avg_valid_g)
        history["valid"]["d"].append(avg_valid_d)
        swanlab.log({
            "epoch": epoch,
            "valid/g_loss": avg_valid_g,
            "valid/d_loss": avg_valid_d,
            "valid/batches": int(valid_n),
        }, step=global_step)
        print(f"[epoch {epoch}][valid] g={avg_valid_g:.4f}  d={avg_valid_d:.4f}  "
              f"(batches: {int(valid_n)})")

        # ---- save best soundstream weights (inference use) ----
        # Rank 0 only.  This is the file you actually load to do generation.
        if avg_valid_g < best_val_loss:
            best_val_loss = avg_valid_g
            best_path = os.path.join(args.ckpt_dir, "best_soundstream.pth")
            torch.save(soundstream.state_dict(), best_path)
            print(f"[epoch {epoch}] new best val_g={best_val_loss:.4f} -> {best_path}")

        # ---- save latest DeepSpeed checkpoint (resume use) ----
        # Includes model + optimizer + scheduler + RNG state.  Collective call
        # but only rank 0 needs the result, so we guard the call itself.
        model_engine.save_checkpoint(
            args.ckpt_dir, "latest",
            client_state={"epoch": epoch, "best_val_loss": best_val_loss},
        )

    dist.barrier()

    # ---- test ----
    model_engine.module.eval()
    test_sampler.set_epoch(epoch)

    local_test_sum_g = 0.0
    local_test_sum_d = 0.0
    local_test_cnt   = 0

    with torch.no_grad():
        for x, lengths_x in tqdm(test_loader, disable=(rank != 0), desc=f"[test  e{epoch}]"):
            x = x.to(device)
            lengths_x = lengths_x.to(device)

            G_x = soundstream(x)

            s_x = torch.stft(x.squeeze(), n_fft=1024, hop_length=256,
                             window=torch.hann_window(window_length=1024, device=device),
                             return_complex=False).permute(0, 3, 1, 2)
            lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")
            s_G_x = torch.stft(G_x.squeeze(), n_fft=1024, hop_length=256,
                               window=torch.hann_window(window_length=1024, device=device),
                               return_complex=False).permute(0, 3, 1, 2)

            lengths_stft = stft_disc.features_lengths(lengths_s_x)
            lengths_wave = wave_disc.features_lengths(lengths_x)

            features_stft_disc_x   = stft_disc(s_x)
            features_wave_disc_x   = wave_disc(x)
            features_stft_disc_G_x = stft_disc(s_G_x)
            features_wave_disc_G_x = wave_disc(G_x)

            loss_g = criterion_g(x, G_x,
                                 features_stft_disc_x, features_wave_disc_x,
                                 features_stft_disc_G_x, features_wave_disc_G_x,
                                 lengths_wave, lengths_stft)

            features_stft_disc_x      = stft_disc(s_x)
            features_wave_disc_x      = wave_disc(x)
            features_stft_disc_G_x_det = stft_disc(s_G_x.detach())
            features_wave_disc_G_x_det = wave_disc(G_x.detach())

            loss_d = criterion_d(features_stft_disc_x, features_wave_disc_x,
                                 features_stft_disc_G_x_det, features_wave_disc_G_x_det,
                                 lengths_stft, lengths_wave)

            local_test_sum_g += loss_g.item()
            local_test_sum_d += loss_d.item()
            local_test_cnt   += 1

    avg_test_g, test_n = all_reduce_mean(local_test_sum_g, local_test_cnt)
    avg_test_d, _      = all_reduce_mean(local_test_sum_d, local_test_cnt)

    dist.barrier()
    if rank == 0:
        history["test"]["g"].append(avg_test_g)
        history["test"]["d"].append(avg_test_d)
        swanlab.log({
            "epoch": epoch,
            "test/g_loss": avg_test_g,
            "test/d_loss": avg_test_d,
            "test/batches": int(test_n),
        }, step=global_step)
        print(f"[epoch {epoch}][test]  g={avg_test_g:.4f}  d={avg_test_d:.4f}  "
              f"(batches: {int(test_n)})")


# ----------------------------------------------------------------------------
# 12. Done.  Only rank 0 dumps the history JSON + closes SwanLab.
# ----------------------------------------------------------------------------
dist.barrier()
if rank == 0:
    hist_path = os.path.join(args.ckpt_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[done] history saved to {hist_path}")
    print(f"[done] best soundstream weights at {os.path.join(args.ckpt_dir, 'best_soundstream.pth')}")
    swanlab.finish()