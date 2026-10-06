# SoundStream 项目模块与类关系梳理

> 项目路径：`/home/luke/distributed_machine_learning/SoundStream/`
> 实现参考：SoundStream 论文 [arXiv:2107.03312](https://arxiv.org/abs/2107.03312)
> RVQ 实现来自：[lucidrains/vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)

> 排版约定：以下 ASCII 图中，每行"显示宽度"严格一致（中文按 2 列、ASCII 按 1 列计）；框的上/下边框与每行的左右 `│`/`┌`/`┐`/`└`/`┘`/`├`/`┤` 完全对齐。

---

## 1. 文件总览

| 文件 | 职责 |
|---|---|
| `main.py` | 训练入口：组装模型 / 加载数据 / 定义损失 / 训练 & 验证 & 测试循环 |
| `net.py` | 所有神经网络模块：`SoundStream` 主体、Encoder、Decoder、Wave Discriminator、STFT Discriminator |
| `dataset.py` | `NSynthDataset`，基于 PyTorch `Dataset` 加载 NSynth `.wav` 文件 |

外部依赖：
- `torch`、`torch.nn`、`torch.nn.functional`
- `torchaudio.transforms.MelSpectrogram`
- `vector_quantize_pytorch.ResidualVQ`

---

## 2. 模块依赖图（文件级）

```
            ┌──────────────────┐
            │   dataset.py     │
            │  NSynthDataset   │
            └────────┬─────────┘
                     │ import
                     ▼
┌──────────────────────────────────────┐
│                main.py               │
│  - collate_fn                        │
│  - adversarial_g_loss / d_loss       │
│  - feature_loss                      │
│  - spectral_reconstruction_loss      │
│  - 训练 / 验证 / 测试 循环            │
└────────┬─────────────────────┬───────┘
         │ import               │ import
         ▼                      ▼
┌─────────────────┐    ┌──────────────────┐
│    net.py       │    │   dataset.py     │
│  SoundStream    │    │  NSynthDataset   │
│  Encoder/Decoder│    └──────────────────┘
│  Discriminators │
└────────┬────────┘
         │ 依赖
         ▼
┌────────────────────────────────────────┐
│ vector_quantize_pytorch.ResidualVQ     │
│  (lucidrains 库,内部用 EuclideanCodebook) │
└────────────────────────────────────────┘
```

---

## 3. 类清单

### 3.1 `net.py`（模型定义）

| 类 | 角色 | 父类 |
|---|---|---|
| `CausalConv1d` | 因果 1D 卷积（左填充） | `nn.Conv1d` |
| `CausalConvTranspose1d` | 因果 1D 转置卷积（右裁剪） | `nn.ConvTranspose1d` |
| `ResidualUnit` | 两层残差块（Conv1d → ELU → Conv1d + skip） | `nn.Module` |
| `EncoderBlock` | Encoder 的一个下采样块（3 个 ResidualUnit + 下采样） | `nn.Module` |
| `DecoderBlock` | Decoder 的一个上采样块（上采样 + 3 个 ResidualUnit） | `nn.Module` |
| `Encoder` | 完整编码器，把波形压成 latent | `nn.Module` |
| `Decoder` | 完整解码器，把 latent 还原成波形 | `nn.Module` |
| `SoundStream` | **生成器**主体（Encoder → ResidualVQ → Decoder） | `nn.Module` |
| `WNConv1d` | 带 weight_norm 的 Conv1d 工厂函数 | — |
| `WaveDiscriminatorBlock` | Wave 判别器的一个 block（含 7 层带权归一化卷积） | `nn.Module` |
| `WaveDiscriminator` | Wave 判别器整体（多尺度） | `nn.Module` |
| `ResidualUnit2d` | 2D 残差块，给 STFT 判别器用 | `nn.Module` |
| `STFTDiscriminator` | STFT 判别器整体 | `nn.Module` |

### 3.2 `dataset.py`（数据加载）

| 类 | 角色 | 父类 |
|---|---|---|
| `NSynthDataset` | 加载 NSynth `.wav`，返回单声道波形 | `torch.utils.data.Dataset` |

### 3.3 `main.py`（训练逻辑）

`main.py` 没有自定义类，只定义：
- 顶层函数 `collate_fn(batch)`：把不同长度的音频 pad 到同长度，返回 `(padded_x, lengths)`
- 顶层函数 `adversarial_g_loss(...)`：生成器对抗损失
- 顶层函数 `feature_loss(...)`：判别器中间层特征匹配损失
- 顶层函数 `spectral_reconstruction_loss(x, G_x)`：多尺度 Mel 频谱重建损失
- 顶层函数 `adversarial_d_loss(...)`：判别器对抗损失
- 顶层 lambda `criterion_g`、`criterion_d`：拼装最终损失

---

## 4. 类的层级与组合关系（UML 类图）

### 4.1 Generator（`SoundStream`）类图

```
┌──────────────────────────────────────────────────┐
│                   SoundStream                    │
│                  (nn.Module)                     │
├──────────────────────────────────────────────────┤
│  encoder    : Encoder                            │
│  quantizer  : ResidualVQ                         │
│  decoder    : Decoder                            │
├──────────────────────────────────────────────────┤
│  forward(x)                                      │
│    e = encoder(x)                                │
│    q = quantizer(e)                              │
│    o = decoder(q)                                │
└─────────────────────┬────────────────────────────┘
                      │
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│           Encoder            │ │           Decoder            │
│         (nn.Module)          │ │         (nn.Module)          │
├──────────────────────────────┤ ├──────────────────────────────┤
│  layers : Sequential         │ │  layers : Sequential         │
├──────────────────────────────┤ ├──────────────────────────────┤
│  CausalConv1d(1,C,k=7)       │ │  CausalConv1d(D,16C,k=7)     │
│  ELU                         │ │  ELU                         │
│  EncoderBlock(out=2C,s=2)    │ │  DecoderBlock(out=8C,s=8)    │
│  ELU                         │ │  ELU                         │
│  EncoderBlock(out=4C,s=4)    │ │  DecoderBlock(out=4C,s=5)    │
│  ELU                         │ │  ELU                         │
│  EncoderBlock(out=8C,s=5)    │ │  DecoderBlock(out=2C,s=4)    │
│  ELU                         │ │  ELU                         │
│  EncoderBlock(out=16C,s=8)   │ │  DecoderBlock(out=C,s=2)     │
│  ELU                         │ │  ELU                         │
│  CausalConv1d(16C,D,k=3)     │ │  CausalConv1d(C,1,k=7)       │
└──────────────────────────────┘ └──────────────────────────────┘
```

#### `EncoderBlock` / `DecoderBlock` 内部结构

```
┌──────────────────────────────┐    ┌──────────────────────────────┐
│        EncoderBlock           │    │           DecoderBlock        │
│        (nn.Module)            │    │           (nn.Module)         │
├──────────────────────────────┤    ├──────────────────────────────┤
│  layers : Sequential         │    │  layers : Sequential         │
│  ├ ResidualUnit(in,out,d=1)  │    │  ├ CausalConvTranspose1d     │
│  ├ ELU                       │    │  ├ ELU                       │
│  ├ ResidualUnit(in,out,d=3)  │    │  ├ ResidualUnit(in,out,d=1)  │
│  ├ ELU                       │    │  ├ ELU                       │
│  ├ ResidualUnit(in,out,d=9)  │    │  ├ ResidualUnit(in,out,d=3)  │
│  ├ ELU                       │    │  ├ ELU                       │
│  └ CausalConv1d(out, k=2s,s) │    │  └ ResidualUnit(in,out,d=9)  │
└──────────────────────────────┘    └──────────────────────────────┘

┌──────────────────────────────┐
│        ResidualUnit          │
│        (nn.Module)           │
├──────────────────────────────┤
│  layers : Sequential         │
│  ├ CausalConv1d(in,out,k=7)  │
│  ├ ELU                       │
│  └ Conv1d(in,out,k=1)        │
├──────────────────────────────┤
│  forward(x) : x + layers(x)  │
└──────────────────────────────┘
```

### 4.2 ResidualVQ 在 SoundStream 中的角色

```
SoundStream.quantizer : ResidualVQ
  └─ layers : List[VectorQuantize]    # 长度 = num_quantizers = n_q
       └─ _codebook : EuclideanCodebook
            ├─ embed          [num_codebooks, codebook_size, dim]
            ├─ cluster_size   [num_codebooks, codebook_size]
            └─ embed_avg      [num_codebooks, codebook_size, dim]
```

### 4.3 Discriminators（判别器）类图

#### WaveDiscriminator

```
┌─────────────────────────────────────────────┐
│              WaveDiscriminator              │
│           (nn.Module, num_D scales)         │
├─────────────────────────────────────────────┤
│  model : ModuleDict[Disc]                   │
│  downsampler : AvgPool1d                    │
├─────────────────────────────────────────────┤
│  forward(x)                                 │
│  features_lengths(lengths)                  │
└─────────────────────┬───────────────────────┘
                      │ 持有 num_D 个
                      ▼
┌─────────────────────────────────────────────┐
│          WaveDiscriminatorBlock             │
│              (nn.Module)                    │
├─────────────────────────────────────────────┤
│  layers : ModuleList (7 层)                 │
│  ├ ReflectionPad1d + WNConv1d(1,16,k=15)    │
│  ├ WNConv1d(16,64,k=41,s=4,g=4)             │
│  ├ WNConv1d(64,256,k=41,s=4,g=16)           │
│  ├ WNConv1d(256,1024,k=41,s=4,g=64)         │
│  ├ WNConv1d(1024,1024,k=41,s=4,g=256)       │
│  ├ WNConv1d(1024,1024,k=5,s=1)              │
│  └ WNConv1d(1024,1,k=3,s=1)                 │
└─────────────────────────────────────────────┘
              每一层后接 LeakyReLU(0.2)
```

#### STFTDiscriminator

```
┌─────────────────────────────────────────────┐
│              STFTDiscriminator              │
│                (nn.Module)                  │
├─────────────────────────────────────────────┤
│  layers : ModuleList (8 层)                 │
│  ├ Conv2d(2,32,k=7)+ELU                     │
│  ├ ResidualUnit2d(32,C,m=2,s_t=1,s_f=2)+ELU │
│  ├ ResidualUnit2d(2C,2C,m=2,s_t=2,s_f=2)+ELU│
│  ├ ResidualUnit2d(4C,4C,m=1,s_t=1,s_f=2)+ELU│
│  ├ ResidualUnit2d(4C,4C,m=2,s_t=2,s_f=2)+ELU│
│  ├ ResidualUnit2d(8C,8C,m=1,s_t=1,s_f=2)+ELU│
│  ├ ResidualUnit2d(8C,8C,m=2,s_t=2,s_f=2)+ELU│
│  └ Conv2d(16C,1,k=(F_bins/64,1))            │
└─────────────────────────────────────────────┘
```

#### ResidualUnit2d

```
┌─────────────────────────────────────────────┐
│              ResidualUnit2d                 │
│                (nn.Module)                  │
├─────────────────────────────────────────────┤
│  layers : Sequential                        │
│  ├ Conv2d(in,N,k=3,padding='same')          │
│  ├ ELU                                      │
│  └ Conv2d(N,mN,k=(s_f+2,s_t+2),s=(s_f,s_t)) │
├─────────────────────────────────────────────┤
│  skip_connection                            │
│  └ Conv2d(in,mN,k=1,s=(s_f,s_t))            │
├─────────────────────────────────────────────┤
│  forward(x): layers(pad(x)) + skip(x)       │
└─────────────────────────────────────────────┘
```

### 4.4 数据集

```
┌─────────────────────────────────────────────┐
│                NSynthDataset                │
│          (torch.utils.data.Dataset)         │
├─────────────────────────────────────────────┤
│  filenames : List[str]                      │
│  sr : int                                   │
├─────────────────────────────────────────────┤
│  __len__()                                  │
│  __getitem__(idx)                           │
│    → torchaudio.load(...)[0]                │
└─────────────────────────────────────────────┘
```

---

## 5. 训练时的实例化关系（`main.py` 的对象图）

```
       train_dataset = NSynthDataset(audio_dir=...)    ◄─── dataset.py
                          │
                          ▼
       train_loader = DataLoader(train_dataset, collate_fn=collate_fn)
                          │
                          │ for x, lengths_x in train_loader:
                          ▼
   ┌──────────────────────────────────────────────────────────┐
   │  x : [B, 1, T_wav]                                       │
   │  lengths_x : [B]                                         │
   └──────────────────────────────────────────────────────────┘
                          │
                          ├────────► stft_disc(s_x)  ──► features_stft_disc_x
                          │
                          ├────────► wave_disc(x)    ──► features_wave_disc_x
                          │
                          ├────────► G_x = soundstream(x)
                          │           │
                          │           ├─► e = soundstream.encoder(x)            ──► Encoder
                          │           │     [B, 1, T_wav]   →  [B, D, T_latent]
                          │           │
                          │           ├─► q = soundstream.quantizer(e)         ──► ResidualVQ
                          │           │     [B, D, T_latent] →  [B, D, T_latent]
                          │           │
                          │           └─► o = soundstream.decoder(q)           ──► Decoder
                          │                 [B, D, T_latent] →  [B, 1, T_wav]
                          │
                          ├────────► features_stft_disc_G_x = stft_disc(s_G_x)
                          ├────────► features_wave_disc_G_x = wave_disc(G_x)
                          │
                          ├────────► loss_g = λ_adv·adversarial_g_loss
                          │           + λ_feat·feature_loss
                          │           + λ_rec ·spectral_reconstruction_loss
                          │
                          └────────► loss_d = adversarial_d_loss(real, generated)
```

### 5.2 三种损失的关键数据流

| 损失 | 输入 | 公式要点 |
|---|---|---|
| `adversarial_g_loss` | `features_stft_disc_G_x`, `features_wave_disc_G_x`, `lengths_*` | `mean( relu(1 − last_layer) )` 在生成样本上 |
| `adversarial_d_loss` | 真实 + 生成样本的判别器特征 | `mean( relu(1 − real) + relu(1 + generated) )` |
| `feature_loss` | 真实 / 生成的判别器中间层特征 | `mean( ‖feat_x − feat_G_x‖₁ )` |
| `spectral_reconstruction_loss` | `x`, `G_x` | 多尺度 Mel 频谱的 L1 + 对数域 L2 |

### 5.3 优化器分工

```
optimizer_g : Adam(soundstream.parameters(),       lr=1e-4, betas=(0.5, 0.9))
optimizer_d : Adam(wave_disc + stft_disc 的参数,   lr=1e-4, betas=(0.5, 0.9))
```

---

## 6. 关键调用时序（一次 train step）

```
1.  x, lengths_x ← next(train_loader)

2.  G_x = soundstream(x)            # Generator 前向
    s_x  = torch.stft(x)            # 真实样本的 STFT
    s_G_x = torch.stft(G_x)         # 生成样本的 STFT

4.  lengths_stft = stft_disc.features_lengths(lengths_s_x)
    lengths_wave = wave_disc.features_lengths(lengths_x)

5.  features_stft_disc_x   = stft_disc(s_x)
    features_wave_disc_x   = wave_disc(x)
    features_stft_disc_G_x = stft_disc(s_G_x)        # 用 G_x（带梯度）
    features_wave_disc_G_x = wave_disc(G_x)

6.  loss_g = criterion_g(x, G_x, features_..., lengths_...)
    loss_g.backward(); optimizer_g.step()

7.  再算一遍 features_stft_disc_x / features_wave_disc_x
    features_stft_disc_G_x_det = stft_disc(s_G_x.detach())
    features_wave_disc_G_x_det = wave_disc(G_x.detach())

8.  loss_d = criterion_d(features_..., features_..._det, lengths_...)
    loss_d.backward(); optimizer_d.step()
```

> 注意：第 7 步必须用 `.detach()`，否则生成样本的梯度会流回 SoundStream，导致 G 和 D 同时被一次 backward 更新。

---

## 7. 形状约定速查

| 阶段 | 形状 | 备注 |
|---|---|---|
| 原始波形 | `[B, 1, T_wav]` | NSynth 4 s × 16 kHz → T_wav = 64000 |
| Encoder 输出 | `[B, D, T_latent]` | T_latent ≈ T_wav / 320 |
| ResidualVQ 输入/输出 | `[B, D, T_latent]` | 注意：库默认 channels-last，需 transpose |
| Decoder 输出 | `[B, 1, T_wav]` | 重建波形 |
| STFT | `[B, 2, F_bins, T_stft]` | 复数拆成实部/虚部 |
| Discriminator 特征 | 多层 list | 每层有自己的时间长度 |

---

## 8. 缺失 / 未实现（README "Missing pieces"）

- **Denoising**：没有条件信号，也没有 FiLM 层
- **Bitrate scalability**：没有 quantizer dropout（虽然库支持 `quantize_dropout`，但 SoundStream 没用）