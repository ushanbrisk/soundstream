# 从单卡到 DeepSpeed DDP：把 SoundStream 改成 6 卡数据并行

> 本文档配套 `main_ddp.py` / `ds_config.json`，所有改动都基于官方 `main.py`。
> 改动原则：**最小改动 + 保留全部原有 loss / 模型结构**。

---

## 0. TL;DR — 启动命令

```bash
# 6 卡 DDP，stage 0 纯数据并行，无 ZeRO
deepspeed --num_gpus=6 main_ddp.py \
    --deepspeed_config ds_config.json

# 断点续训
deepspeed --num_gpus=6 main_ddp.py \
    --deepspeed_config ds_config.json --resume
```

启动后产物：
```
checkpoints/
├── best_soundstream.pth        # 仅 soundstream 权重，纯推理用，rank 0 写
├── history.json                # 训练/验证/测试 loss 曲线，rank 0 写
└── latest/                     # DeepSpeed 完整 ckpt（model + opt + sched + rng）
    ├── latest_tag.txt
    ├── mp_rank_00_*.pt ... mp_rank_05_*.pt  # 6 个 rank 各一份
    └── ...
```

---

## 1. 整体改动一览

| 改动点 | 原 `main.py` | 新 `main_ddp.py` | 原因 |
|------|-----------|----------------|------|
| 启动方式 | `python main.py` | `deepspeed --num_gpus=6 main_ddp.py` | DeepSpeed launcher 起 6 个进程 |
| 模型持有方式 | 三个独立 `nn.Module` + 两个独立 optimizer | `JointModel` 包成一个 `multi_modulenn.Module`，DeepSpeed 单 engine | DeepSpeed.initialize 只接受一个 model |
| Optimizer | `optim.Adam(...) ×2` | `deepspeed.initialize()` 返回的 engine | 由 ds_config.json 统一配置 |
| 反向传播 | `loss.backward(); optimizer.step()` | `model_engine.backward(loss); model_engine.step()` | 适配 DDP all-reduce |
| 数据集 | `DataLoader(...)` | `DataLoader(..., sampler=DistributedSampler(...))` | 6 个 rank 各取一份数据 |
| 数据 shuffle | DataLoader 内部 | `sampler.set_epoch(epoch)` | DDP 下 shuffle 由 epoch 决定 |
| 验证/test loss 聚合 | 单进程累加 | 每 rank 局部累加 → `dist.all_reduce` 求全局平均 | 6 个 rank 各看一份数据 |
| 权重保存 | ❌ 没保存 | `torch.save(soundstream.state_dict(), ...)` + `model_engine.save_checkpoint(...)` | 补上保存逻辑 |
| 文件 I/O 守卫 | 无 | `if rank == 0:` + `dist.barrier()` | 避免 6 个 rank 同时写同一个文件 |

**没改的**：4 个 loss 函数（`adversarial_g_loss` / `feature_loss` / `spectral_reconstruction_loss` / `adversarial_d_loss`）、`net.py` / `dataset.py`、超参数 `LAMBDA_*`。

---

## 2. 文件改动逐项说明

### 2.1 新增 `ds_config.json`

```json
{
  "train_batch_size": 24,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 1.0,
  "optimizer": {
    "type": "Adam",
    "params": { "lr": 1e-4, "betas": [0.5, 0.9] }
  },
  "zero_optimization": { "stage": 0 },
  "steps_per_print": 100
}
```

要点：
- **`train_batch_size` 必须给数字**，DeepSpeed 0.15.4 这条老版本不识别 `"auto"`（会报 `TypeError: '>' not supported between instances of 'str' and 'int'`）。如果你想用 `"auto"`，升级 DeepSpeed 或自己写 `train_batch_size = world_size × micro × grad_accum`。
- `train_batch_size: 24 = 6 GPUs × 4 batch × 1 grad_accum`。**改了 GPU 数量或 batch，要手动改这个值**，否则 DeepSpeed 会按错的总数切 train。
- `train_micro_batch_size_per_gpu: 4` → 每张卡 micro_batch=4。**注意这是单卡 batch**，原 `main.py` 里 `BATCH_SIZE=4` 在 DDP 下也是单卡 4。
- `stage: 0` → 纯 DDP，不切 ZeRO。如果你以后想上 ZeRO-2 省显存，把这里改成 2 即可，其它代码不用动。
- **没有 `fp16` / `bf16` 配置**，与原始 `main.py` 保持一致（纯 fp32）。如果你以后想开混合精度，往里加：
  ```json
  "bf16": { "enabled": true }
  ```
  注意 `fp16` 容易数值塌陷（loss=nan），`bf16` 更稳（同精度，无需 fp16）；A100/A10/3090/3080 都支持 bf16。

---

### 2.2 `main.py:1-13`（imports）

**原**：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**新**（`main_ddp.py:21-37`）：
```python
import argparse
import json
import os

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
...

# 1. CLI args.  --local_rank is required so DeepSpeed's launcher can inject it.
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, ...)
parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
parser.add_argument("--resume", action="store_true", ...)
args = parser.parse_args()

# 2. Distributed init.  Must come BEFORE any per-rank setup that touches CUDA.
deepspeed.init_distributed()
local_rank = int(os.environ["LOCAL_RANK"])
rank       = dist.get_rank()
world_size = dist.get_world_size()
device     = torch.device(f"cuda:{local_rank}")
```

要点：
- **`--local_rank` 必须存在**，DeepSpeed launcher 靠它识别每个进程对应的 GPU。
- **`deepspeed.init_distributed()` 必须在任何 CUDA 操作之前**。它等价于 `torch.distributed.init_process_group(backend="nccl")` + 一些额外设置。
- **不要**在这里加 `if rank == 0: ...` 早返回——6 个进程都得走完这段，否则 `dist.barrier()` 会卡死。

---

### 2.3 `main.py:21-28`（3 个模型 + 2 个 optimizer）

**原**：
```python
soundstream = SoundStream(C=32, D=128, n_q=8, codebook_size=1024)
wave_disc = WaveDiscriminator(num_D=3, downsampling_factor=2)
stft_disc = STFTDiscriminator(C=1, F_bins=W//2)

soundstream.to(device); wave_disc.to(device); stft_disc.to(device)
...
optimizer_g = optim.Adam(soundstream.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizer_d = optim.Adam(list(wave_disc.parameters()) + list(stft_disc.parameters()),
                        lr=1e-4, betas=(0.5, 0.9))
```

**新**（`main_ddp.py:79-99`）：
```python
class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.soundstream = SoundStream(C=32, D=128, n_q=8, codebook_size=1024)
        self.wave_disc   = WaveDiscriminator(num_D=3, downsampling_factor=2)
        self.stft_disc   = STFTDiscriminator(C=1, F_bins=1024 // 2)

joint_model = JointModel()

# 让 deepspeed.initialize() 自己读 ds_config.json（路径来自 argparse 的
# --deepspeed_config 标志）。如果同时传 config= 和 args.deepspeed_config，
# DeepSpeed 会抛 AssertionError：
#   Not sure how to proceed, we were given deepspeed configs in the deepspeed
#   arguments and deepspeed.initialize() function call
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
```

要点：
- **为什么用一个 JointModel 包三个？**
  `deepspeed.initialize()` 一次只能接受一个 `nn.Module`。三个网络只能包成一个。
- **为什么一个 optimizer 同时管 G 和 D？**
  原代码有两个 Adam。原 G step 只对 G 的参数产生梯度（loss_g 不经过 D），原 D step 只对 D 的参数产生梯度（loss_d 那一侧 detach 了）。DeepSpeed 的 optimizer 在 `.step()` 时只更新有 `.grad != None` 的参数，所以一个 Adam 足够。
- **不要显式 `.to(device)`**——DeepSpeed 会自动放到当前 rank 对应的 GPU。
- **`model_engine.module.X` vs `model_engine.X`**：在 stage 0 下两者等价；但 stage > 0 时参数可能被打散到不同 rank，必须走 `.module`。

---

### 2.4 `main.py:34-42`（3 个 DataLoader）

**原**：
```python
train_dataset = NSynthDataset(audio_dir="/M101/dataset/audio/NSynth/nsynth-train/audio")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=2)
...
valid_loader = DataLoader(valid_dataset, ...)
test_loader  = DataLoader(test_dataset, ...)
```

**新**（`main_ddp.py:104-122`）：
```python
train_dataset = NSynthDataset(audio_dir="/M101/dataset/audio/NSynth/nsynth-train/audio")
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                           collate_fn=collate_fn, num_workers=2)

valid_dataset = NSynthDataset(audio_dir="/M101/dataset/audio/NSynth/nsynth-valid/audio")
valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
valid_loader  = DataLoader(valid_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler,
                           collate_fn=collate_fn, num_workers=2)

test_dataset = NSynthDataset(audio_dir="/M101/dataset/audio/NSynth/nsynth-test/audio")
test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler,
                          collate_fn=collate_fn, num_workers=2)
```

要点：
- **`sampler` 和 `shuffle=True` 不能同时给**。给 sampler 时 DataLoader 内部就不 shuffle 了。
- **shuffle 的随机性靠 `sampler.set_epoch(epoch)`**（见下面训练循环），否则每个 epoch 每个 rank 看到的顺序一样。
- **`batch_size` 的语义变了**：原来是"全局 batch_size"，现在是"**单卡**的 micro_batch_size"。如果原来 `BATCH_SIZE=4` 想保持全局 4，就把这里改成 1（但太小不建议）；如果想全局 24（6×4），每张卡还是 4，正好对得上。

---

### 2.5 训练循环（`main.py:104-150` → `main_ddp.py:191-260`）

核心替换清单：

| 位置 | 原写法 | 新写法 |
|------|------|------|
| 模型调用 | `soundstream(x)` | `soundstream(x)` （变量名沿用，已经指向 module） |
| G 反向 | `loss_g.backward()` | `model_engine.backward(loss_g)` |
| G optimizer | `optimizer_g.zero_grad(); optimizer_g.step()` | `model_engine.zero_grad(); model_engine.step()` |
| D 反向 | `loss_d.backward()` | `model_engine.backward(loss_d)` |
| D optimizer | `optimizer_d.zero_grad(); optimizer_d.step()` | `model_engine.zero_grad(); model_engine.step()` |
| tqdm | `tqdm(train_loader)` | `tqdm(train_loader, disable=(rank != 0))` |

为什么 `disable=(rank != 0)`：6 个 rank 同时打 progress bar 会把终端刷爆，只让 rank 0 显示。

---

### 2.6 验证 / 测试 loss 聚合（`main.py:159-191` → `main_ddp.py:262-378`）

**原**：
```python
valid_loss_g = 0.0
for x, lengths_x in tqdm(valid_loader):
    ...
    valid_loss_g += loss_g.item()
history["valid"]["g"].append(valid_loss_g / len(valid_loader))
```

**问题**：DDP 下每个 rank 只看到 1/world_size 的数据。如果直接累加再除 `len(valid_loader)`，得到的只是**该 rank 的局部平均**，不是全局平均。

**新**（`main_ddp.py:169-179` 的辅助函数 + 训练循环里的调用）：
```python
def all_reduce_mean(local_sum: float, local_count: int):
    """Return (global_mean, global_count).  Works from every rank."""
    t = torch.tensor([local_sum, local_count], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t[0] / t[1]).item(), t[1].item()
```

训练/验证/test 循环里：
```python
local_valid_sum_g += loss_g.item()
local_valid_sum_d += loss_d.item()
local_valid_cnt   += 1
...
avg_valid_g, valid_n = all_reduce_mean(local_valid_sum_g, local_valid_cnt)
avg_valid_d, _       = all_reduce_mean(local_valid_sum_d, local_valid_cnt)
```

要点：
- `dist.all_reduce` 是 collective 操作，**每个 rank 都要调用**，而且每个 rank 拿到的 local 数据可以不同。
- 这里故意传 `[sum, count]` 而不是分别 all_reduce 再除——一次 all_reduce 更省一次通信。
- 用 `float64` 是避免累加大量小数值时的精度损失。

---

### 2.7 权重保存（**新增**，原 main.py 没保存）

`main_ddp.py:305-323`：

```python
if rank == 0:
    # ---- save best soundstream weights (inference use) ----
    if avg_valid_g < best_val_loss:
        best_val_loss = avg_valid_g
        best_path = os.path.join(args.ckpt_dir, "best_soundstream.pth")
        torch.save(soundstream.state_dict(), best_path)
        print(f"[epoch {epoch}] new best val_g={best_val_loss:.4f} -> {best_path}")

    # ---- save latest DeepSpeed checkpoint (resume use) ----
    model_engine.save_checkpoint(
        args.ckpt_dir, "latest",
        client_state={"epoch": epoch, "best_val_loss": best_val_loss},
    )

dist.barrier()
```

两种保存方式的区别：

| | `best_soundstream.pth` | `checkpoints/latest/` |
|---|---|---|
| 内容 | 仅 `soundstream.state_dict()` | model + optimizer + scheduler + RNG + `client_state` |
| 用途 | 推理 / 导出 / 评估 | 续训（`--resume`） |
| 谁能加载 | 任何 PyTorch | DeepSpeed engine 专属 |
| 谁写 | rank 0 | 6 个 rank 各自写自己那一份 |

`client_state` 是 DeepSpeed 提供的"用户自定义状态"，我们塞了 `epoch` 和 `best_val_loss`，这样 resume 时能直接读回来。

---

### 2.8 `main.py:95` 处的 `best_model` 内存副本

**原**：
```python
best_model = soundstream.state_dict().copy()  # 仅内存，进程退出就丢
```

**问题**：脚本结束后 `best_model` 跟着 Python 进程一起死，下次想推理还得重训。

**新**：彻底删掉这行内存备份，改为每 epoch 写到 `best_soundstream.pth`（见 2.7）。同一个 epoch 之内不需要保留内存备份，因为每 epoch 都会重判。

---

### 2.9 续训（resume）

`main_ddp.py:147-159`：

```python
if args.resume:
    _, client_state = model_engine.load_checkpoint(args.ckpt_dir, "latest")
    if client_state is not None:
        start_epoch   = client_state.get("epoch", 0) + 1
        best_val_loss = client_state.get("best_val_loss", float("inf"))
    if rank == 0:
        print(f"[resume] starting from epoch {start_epoch}, best_val_loss={best_val_loss}")
```

要点：
- `load_checkpoint` **每个 rank 都要调用**（它是 collective），否则会卡住或报错。
- 第一个返回值是 `success: bool`，我们用 `_` 丢掉；如果 `client_state` 是 None，说明根本没 ckpt，按全新训练走。

---

### 2.10 目录创建（rank 0 单独建，别的 barrier 等）

`main_ddp.py:141-144`：
```python
if rank == 0:
    os.makedirs(args.ckpt_dir, exist_ok=True)
dist.barrier()
```

理由：6 个 rank 同时调 `os.makedirs(..., exist_ok=True)` 倒也不会报错（exist_ok=True 是原子的），但写日志 / 写检查点会互相踩，所以统一 rank 0 干、其它 barrier 等。

---

## 3. 关于"rank 才进行操作"的判定准则

什么时候文件 I/O 必须 rank 0？把所有 I/O 分类：

| 操作 | 必须 rank 0？ | 原因 |
|------|--------------|------|
| `os.makedirs()` | ✅ | 6 个进程同时建同目录 + 同时往里写 race |
| `torch.save(model_weights, ...)` | ✅ | 6 份重复文件，浪费 IO |
| `model_engine.save_checkpoint()` | ⚠️ 内部已处理 | DeepSpeed 会让每个 rank 写自己的 `mp_rank_XX_*.pt`，但只要从 rank 0 触发一次 |
| `model_engine.load_checkpoint()` | ❌ 全员 | 是 collective，少一个 rank 直接 hang |
| `print()` | 建议 ✅ | 6 份日志没人看 |
| `tqdm()` | 建议 ✅ | 终端被刷爆 |
| `loss.item()` / 算 loss | ❌ 全员 | 每个 rank 都得算自己的那份 |
| `dist.all_reduce()` | ❌ 全员 | 是 collective |

判断口诀：**写文件 = rank 0；动 collective = 全员**。

---

## 4. 常见坑

### 4.1 `if rank == 0: return` 在 `deepspeed.init_distributed()` 之前
会卡死。`init_distributed` 必须在每个 rank 都执行。

### 4.2 `argparse` 又没加 `--local_rank`
DeepSpeed launcher 启动时会注入这个 env var，但 argparse 也得有这个 flag，否则 `deepspeed.initialize(args=args, ...)` 会报 unexpected keyword。

### 4.3 `loss.item()` 阻断反向传播
不影响，因为只在 `.item()` 时才把 tensor 拷回 CPU，反向图已经走完了。

### 4.4 没调 `sampler.set_epoch(epoch)`
每个 epoch 各 rank 看到的样本顺序一样，等于没 shuffle。一定要在 epoch 开头调。

### 4.5 `num_workers=2` × 6 卡 = 12 个 worker 进程
如果你机器只有 6 核 CPU，可能 IO 反而更慢。可以改成 `num_workers=0` 试试。

### 4.6 `history` 字典
原来只在单进程用，每个 epoch 写一次。在 DDP 下每个 rank 都会经过这段，所以 **rank != 0 的 process 必须不能写 history**，否则写文件 race。`main_ddp.py:163-170` 用 `if rank == 0:` 包了整个 history 的初始化，loop 里只在 rank 0 `history[...]append()`。

### 4.7 混合精度（fp16 / bf16）
当前 `ds_config.json` 不开混合精度，与原 `main.py` 一致（纯 fp32）。如果你以后开了 fp16 / bf16，注意 `loss` 已经是 scaled，`model_engine.backward(loss)` 会自动 unscaled。**不要再手动 `loss.backward()`**，否则梯度会按 scaled 的版本更新。

### 4.8 ZeRO > 0 时的 `.state_dict()`
`torch.save(soundstream.state_dict(), ...)` 在 ZeRO-2/3 下只保存了当前 rank 那一份参数切片（不完整），要换用 `model_engine.save_checkpoint()`。当前我们 stage 0，没踩这个坑。

---

## 5. 改动量统计

| 项 | 数量 |
|----|------|
| 新增文件 | 2（`main_ddp.py`、`ds_config.json`） |
| 新增文档 | 1（`tutorial/ddp.md`，本文档） |
| 修改 `main.py` | 0（保持原样） |
| `main_ddp.py` 总行数 | ~378 |

`main.py` 一行未动，可以直接 `diff main.py main_ddp.py` 看完整差异。

---

## 6. 异构 GPU 训练（3×3090 + 3×3080 之类）

**结论：能跑，但要为最弱那张卡服务**。

| 卡型 | 显存 | 算力 |
|------|------|------|
| RTX 3090 | 24 GB | 8.6 |
| RTX 3080 | 10 GB | 8.6 |

3090 是 3080 的 ~2.4 倍显存，但两者算力 SM 数差不了太多。DDP 下 6 张卡各跑各的 batch，**全部要等到最慢/最先 OOM 的那张卡结束才同步**，所以：
- 3090 经常跑不满（受 3080 batch 大小限制）
- 3080 容易 OOM（batch 不小心调大就崩）

### 三种应对方案

#### 方案 A：只跑 3 张 3090（最省心）

```bash
CUDA_VISIBLE_DEVICES=0,1,2 deepspeed --num_gpus=3 main_ddp.py
```
24GB 跑 fp32 + batch=4 没问题。**推荐先用这个验证流程**，确认流程无误后再扩展到全部 6 张。

#### 方案 B：6 张全用，缩 batch + grad_accumulation

3080 在 fp32 + batch=4 下大概率 OOM。先把 ds_config 里的 micro batch 调到 1，再开 gradient accumulation 维持有效 batch：

```json
{
  "train_batch_size": 24,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 4
}
```
等效于：每张卡 micro_batch=1，累 4 次再 step，**全局有效 batch = 6×1×4 = 24**（同原来 6×4=24）。

但需要注意：**3090 在 batch=1 下也很闲**，算力利用率上不去。这一方案适合"显存是瓶颈"但"算力充裕"的训练。

#### 方案 C：6 张全用 + 3090/3080 各跑合适的 batch

> 这条路**比较脏**，需要自己用 `--train_micro_batch_size_per_gpu` 命令行参数覆盖，而且 DeepSpeed 当前不支持"按 rank 给不同 micro batch"，所以本质上还是**统一 batch**，只是挑一个所有卡都能扛的值。

实际做法：

```bash
# 先开小一点（每卡 batch=2，3090 跑不满但 3080 能扛）
deepspeed --num_gpus=6 main_ddp.py \
    --deepspeed_config ds_config.json \
    --train_micro_batch_size_per_gpu 2
```

或者干脆**两个 GPU 类型分开训练两个实验**，比异构 batch 省心：

| 实验 | 卡 | 命令 |
|---|---|---|
| 实验 A | 3×3090 | `CUDA_VISIBLE_DEVICES=0,1,2 deepspeed --num_gpus=3 main_ddp.py` |
| 实验 B | 3×3080 | `CUDA_VISIBLE_DEVICES=3,4,5 deepspeed --num_gpus=3 main_ddp.py --train_micro_batch_size_per_gpu 1` |

如果你真的想榨干 3090 同时让 3080 不 OOM，更靠谱的路是 **上 ZeRO-3**（`stage: 3`），模型参数本身被切碎，单卡不需要装下整个模型，3080 也能扛大 micro batch。但这要业务代码额外配合 `model_engine.module.gather_params()` 之类的接口，**改动比 stage 0 大**，建议先按方案 A/B 跑通再说。

### 推荐起步步骤

1. **先方案 A（3×3090）跑通**，1 个 epoch 内能正常出 loss 即可。
2. 接着**试 6 张全用 + batch=2**（`--train_micro_batch_size_per_gpu 2`），看 3080 是否 OOM：跑出 OOM 就降到 batch=1。
3. 流程稳了之后，再考虑**方案 C 的 ZeRO-3 改造**。