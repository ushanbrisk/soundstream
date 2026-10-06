# SoundStream 训练框架总览（main.py）

> 项目路径：`/home/luke/distributed_machine_learning/SoundStream/`
> 参考实现：[SoundStream (arXiv:2107.03312)](https://arxiv.org/abs/2107.03312)
> 对应代码：`main.py` + `net.py`
>
> 排版约定与 [`class.md`](./class.md) 一致 —— ASCII 框图的"显示宽度"严格对齐
> （中文按 2 列、ASCII 按 1 列计）。损失函数使用 LaTeX 数学公式。

---

## 1. 三秒总览

整个系统本质上是一个 **条件 GAN**：生成器 `SoundStream` 把波形 `x` 编-解-码出重建波形 `G_x`，两个判别器
`wave_disc`（时域）和 `stft_disc`（频域）负责区分真 / 假波形。生成器用三组损失的加权和优化：

$$
\mathcal{L}_G \;=\; \lambda_{\text{adv}}\,\mathcal{L}_{\text{adv}}^{G}
\;+\; \lambda_{\text{feat}}\,\mathcal{L}_{\text{feat}}
\;+\; \lambda_{\text{rec}}\,\mathcal{L}_{\text{rec}}
\quad(\lambda_{\text{adv}}=1,\;\lambda_{\text{feat}}=100,\;\lambda_{\text{rec}}=1)
$$

判别器只优化 hinge 形式的对抗损失 $\mathcal{L}_{\text{adv}}^{D}$。

---

## 2. 顶层数据流框图（一张图看完整个 forward / backward）

```
                          ┌───────────────────────────────────────────┐
                          │            main.py 训练循环               │
                          └───────────────────────────────────────────┘

  ┌──────────┐
  │ 真实波形 │        ┌─────────────────────────────────────────────┐
  │    x     │───────▶│  SoundStream    G(·)                        │
  │ (B,1,T)  │        │  ┌─────────┐   ┌────────┐   ┌─────────┐    │
  └────┬─────┘        │  │ Encoder │──▶│Residual│──▶│ Decoder │    │
       │              │  │  E(·)   │   │  VQ    │   │  D(·)   │    │
       │              │  └─────────┘   └────────┘   └─────────┘    │
       │              └──────────────┬──────────────────────────────┘
       │                             │ G_x  (重建波形, 同 shape)
       │                             ▼
       │            ┌────────────────────────────────────────────────┐
       │  ────────▶  │  STFT 预处理：s = STFT(x), s_G = STFT(G_x)      │
       │            │  n_fft=1024, hop=256, hann window, 2 通道实/虚 │
       │            └────────────────────────────────────────────────┘
       │                │                                  │
       │                ▼                                  ▼
       │      ┌───────────────────┐              ┌──────────────────────┐
       │      │   stft_disc       │              │   wave_disc          │
       │      │   S(·)            │              │   W(·)               │
       │      │ 2D Conv on (F,T)  │              │ 1D Conv on waveform  │
       │      │ 返回 8 个 feature │              │ 3 个尺度：disc_1/2/4 │
       │      │ (最后 1 个是 logits)│              │ 每个返回 7 个 feature│
       │      └─────────┬─────────┘              └──────────┬───────────┘
       │                │ features_stft_disc_x              │ features_wave_disc_x
       │                │ features_stft_disc_G_x            │ features_wave_disc_G_x
       │                ▼                                  ▼
       │      ┌──────────────────────────────────────────────────────────┐
       │      │                四 组 损 失 计 算                          │
       │      │                                                          │
       │      │  ┌──────────────────────┐  ┌──────────────────────┐      │
       │      │  │ adversarial_g_loss   │  │ adversarial_d_loss   │      │
       │      │  │ L_adv^G              │  │ L_adv^D              │      │
       │      │  └──────────────────────┘  └──────────────────────┘      │
       │      │  ┌──────────────────────┐  ┌──────────────────────┐      │
       │      │  │ feature_loss         │  │ spectral_reconstruction│    │
       │      │  │ L_feat (中间特征匹配) │  │ _loss L_rec (多尺度梅尔)│   │
       │      │  └──────────────────────┘  └──────────────────────┘      │
       │      └─────────────────────┬────────────────────┬──────────────┘
       │                            ▼                    ▼
       │              ┌────────────────────┐  ┌────────────────────┐
       │              │ 优化器 optimizer_g │  │ 优化器 optimizer_d │
       │              │ 更新 SoundStream   │  │ 更新 wave_disc     │
       │              │                    │  │ + stft_disc        │
       │              └────────────────────┘  └────────────────────┘
       ▼
   (再走一遍评估循环 valid / test)
```

> 关键约定：判别器既返回**最后层 logits**（用于对抗损失），也返回**所有中间层 feature maps**
> （用于 feature matching 损失）。同一个 forward 调用拿到两组信息（详见 `WaveDiscriminatorBlock.forward` 与
> `STFTDiscriminator.forward`，两者都用 `feature_map = []` 在循环中累积）。

---

## 3. 每个"框"的输入 / 输出（细化）

### 3.1 `SoundStream` 框（生成器 G）

```
   ┌──────────────────────────── SoundStream (C=32, D=128, n_q=8, codebook_size=1024) ────────────────────────────┐
   │                                                                                                              │
   │   x ─▶ [ Encoder (1→C→2C→4C→8C→16C→D, CausalConv1d + ResidualUnit + 下采样) ]                              │
   │              │                          │                                                                  │
   │              ▼                          ▼                                                                  │
   │          e (B,D,T_enc) ─▶ trans (B,T_enc,D) ─▶ [ ResidualVQ (8 个 codebook 残差量化) ]                     │
   │                                                                       │                                    │
   │                                                                       ▼                                    │
   │                                            quantized (B,D,T_enc) ─▶ trans (B,D,T_enc)                       │
   │                                                                       │                                    │
   │                                                                       ▼                                    │
   │                          [ Decoder (D→16C→8C→4C→2C→C→1, CausalConvTranspose1d + ResidualUnit + 上采样) ]   │
   │                                                                       │                                    │
   │                                                                       ▼                                    │
   │                                                                    G_x (B,1,T)                              │
   └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

| 端口 | 含义 | 张量形状 |
|---|---|---|
| **输入** `x` | 真实音频波形（已 pad 到 batch 内最大长度） | `(B, 1, T)` |
| **内部** `e` | encoder 输出 / VQ 输入 | `(B, D, T_enc)` |
| **内部** `quantized` | RVQ 量化结果（直通估计器） | `(B, D, T_enc)` |
| **输出** `G_x` | 重建波形 | `(B, 1, T)` |

* Encoder 下采样步幅 `2×4×5×8 = 320`，所以 `T_enc = T / 320`。
* ResidualVQ 包含 8 个量化器，每个 codebook 大小 1024，因此总码率 ≈ `8 × log2(1024) / T × sr` bit/s。

### 3.2 `wave_disc` 框（多尺度时域判别器 W）

```
                       ┌────────────── WaveDiscriminator (num_D=3, downsampling_factor=2) ──────────────┐
                       │                                                                                │
   x (B,1,T) ─▶       │                                                                            │
                       │   ┌── AvgPool1d(k=4,s=2) ─▶ x' (B,1,T/2) ─▶                                  │
                       │   │                                                                       │
                       │   ▼                                                                       │
   disc_1 (B,1,T)  ──▶ ┌──────────────────────────┐   ┌──────────────────────────┐   ┌────────────────────────┐
                       │ WaveDiscriminatorBlock   │   │ WaveDiscriminatorBlock   │   │ WaveDiscriminatorBlock  │
                       │ (raw)                    │   │ (disc_2, x' downsample)  │   │ (disc_4, x'' downsample)│
                       │                          │   │                          │   │                        │
                       │ ReflectionPad → WNConv1d │   │ 同左                      │   │ 同左                    │
                       │ 1→16→64→256→1024→1024    │   │                          │   │                        │
                       │ →WNConv1d→1              │   │                          │   │                        │
                       │                          │   │                          │   │                        │
                       │ 返回 7 个 feature_map    │   │ 返回 7 个 feature_map    │   │ 返回 7 个 feature_map   │
                       │ 最后 1 个是 logits       │   │ 最后 1 个是 logits       │   │ 最后 1 个是 logits      │
                       └──────────────────────────┘   └──────────────────────────┘   └────────────────────────┘
```

| 端口 | 含义 | 形状 |
|---|---|---|
| **输入** `x` | 真实 / 生成波形 | `(B, 1, T)` |
| **输出** `features_wave_disc_x` | 字典 `{disc_1, disc_2, disc_4} → list[7 feature maps]`，最后一层是 logits | `dict[str, list[Tensor]]` |

### 3.3 `stft_disc` 框（频域判别器 S）

```
   ┌───────────────────────────────────── STFTDiscriminator (C=1, F_bins=W/2=512) ─────────────────────────────────┐
   │                                                                                                              │
   │   s (B,2,F,T_stft) ─▶ [ Conv2d 2→32 k=(7,7) ]                                                                │
   │                              │                                                                               │
   │                              ▼                                                                               │
   │                6× ResidualUnit2d (2D 卷积 + 残差 + 下采样)                                                    │
   │                              │                                                                               │
   │                              ▼                                                                               │
   │                     [ Conv2d 16C→1 k=(F_bins/64, 1) ]                                                        │
   │                              │                                                                               │
   │                              ▼                                                                               │
   │                       logits (B,1,1,T_stft)                                                                   │
   │                                                                                                              │
   │   整个 forward 同时把 8 个 feature_map 收集起来返回                                                           │
   └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

| 端口 | 含义 | 形状 |
|---|---|---|
| **输入** `s` | `torch.stft(..., return_complex=False)`，最后一维是实/虚 2 通道，再 `permute(0,3,1,2)` | `(B, 2, F, T_stft)` |
| **输出** `features_stft_disc_x` | `list[8]`，最后一层是 logits | `list[Tensor]` |

> 注意：`stft_disc` 在波形进入前需要先经 **STFT 预处理**（在 `main.py` 训练循环里手动完成，见第 2 节），
> 而不是封装在网络内部。

---

## 4. 各 Loss 的精确定义（公式 + 代码对应）

为方便对照，下面记号约定如下：

| 符号 | 含义 |
|---|---|
| $x$ | 真实波形 |
| $G_x = G(x)$ | 重建波形 |
| $s_x = \text{STFT}(x)$, $s_{G_x} = \text{STFT}(G_x)$ | 真实 / 生成频域张量 |
| $S_l(x), S_l(G_x)$ | 第 $l$ 尺度 Mel 频谱（$s = 2^l$, $l=6..11$） |
| $W^{(k)}_l, S_l$ | 第 $k$ 个 wave 子判别器 / stft_disc 第 $l$ 层 feature map |
| $D^W, D^S$ | $W$ 与 $S$ 的最后一层 logits |
| $\ell^W_l, \ell^S_l$ | 对应 feature map 的"有效长度"（用来做长度归一化） |
| $\lambda_{\text{adv}}=1,\ \lambda_{\text{feat}}=100,\ \lambda_{\text{rec}}=1$ | 损失权重 |

---

### 4.1 生成器对抗损失  $\mathcal{L}_{\text{adv}}^{G}$（hinge 形式）

$$
\mathcal{L}_{\text{adv}}^{G}
\;=\;
\mathbb{E}\!\left[\operatorname{relu}\!\left(1 - D^{S}(s_{G_x})\right)\right]
\;+\;
\sum_{k \in \{1,2,4\}}
\mathbb{E}\!\left[\operatorname{relu}\!\left(1 - D^{W^{(k)}}(G_x)\right)\right]
$$

代码（`adversarial_g_loss`）：

```python
stft_loss = F.relu(1 - features_stft_disc_G_x[-1]).sum(dim=3).squeeze() / lengths_stft[-1].squeeze()
wave_loss = torch.cat([
    F.relu(1 - features_wave_disc_G_x[k][-1]).sum(dim=2).squeeze() / lengths_wave[k][-1].squeeze()
    for k in wave_disc_names
])
loss = torch.cat([stft_loss, wave_loss]).mean()
```

> 这里的 `.sum(...).squeeze() / lengths` 是把 logits 在时间维求和后除以有效长度，等价于对有效样本做均值；
> `F.relu(1 - D(G_x))` 就是标准 hinge 生成器目标 $\min_D \max_G \mathbb{E}[\operatorname{relu}(1-D(\cdot))]$。

---

### 4.2 特征匹配损失  $\mathcal{L}_{\text{feat}}$（feature matching / perceptual loss）

把每个判别器在每一中间层抽出的 feature map 当成"感知特征"，要求真实 / 生成的特征尽量一致：

$$
\mathcal{L}_{\text{feat}}
=
\frac{1}{L_S}\sum_{l=0}^{L_S-1}
\frac{\|S_l(x) - S_l(G_x)\|_1}{\ell^S_l}
\;+\;
\frac{1}{L_W}\sum_{k}\sum_{l=0}^{L_W-1}
\frac{\|W^{(k)}_l(x) - W^{(k)}_l(G_x)\|_1}{\ell^W_{k,l}}
$$

代码（`feature_loss`）：

```python
stft_loss = torch.stack([
    ((feat_x - feat_G_x).abs().sum(dim=-1) / lengths_stft[i].view(-1,1,1))
        .sum(dim=-1).sum(dim=-1)
    for i, (feat_x, feat_G_x) in enumerate(zip(features_stft_disc_x, features_stft_disc_G_x))
], dim=1).mean(dim=1, keepdim=True)

wave_loss = torch.stack([
    torch.stack([
        (feat_x - feat_G_x).abs().sum(dim=-1).sum(dim=-1) / lengths_wave[k][i]
        for i, (feat_x, feat_G_x) in enumerate(zip(features_wave_disc_x[k], features_wave_disc_G_x[k]))
    ], dim=1)
    for k in wave_disc_names
], dim=2).mean(dim=1)

loss = torch.cat([stft_loss, wave_loss], dim=1).mean()
```

> 特点：
> - **不取最后一层 logits**（最后一层只用来算 $\mathcal{L}_{\text{adv}}$）。
> - L1 距离 + 长度归一化，对变长 batch 友好。

---

### 4.3 多尺度梅尔谱重建损失  $\mathcal{L}_{\text{rec}}$

跨 6 个不同时间分辨率的 MelSpectrogram 上同时算 L1 + 加权 spectral convergence：

$$
\mathcal{L}_{\text{rec}}
=
\sum_{l=6}^{11}\Bigg[
\underbrace{\|S_l(x) - S_l(G_x)\|_1}_{\text{log-magnitude L1}}
\;+\;
\alpha_l \cdot \underbrace{\sqrt{\sum_{f}\!\big(\log S_l(x)_f - \log S_l(G_x)_f\big)^{\!2}}}_{\text{spectral convergence}}
\Bigg]
$$

其中 $\alpha_l = \sqrt{s_l/2}$, $s_l = 2^l$，窗长 $s_l$，hop $= s_l/4$，`n_mels=8`。

代码（`spectral_reconstruction_loss`）：

```python
L = 0
for i in range(6, 12):
    s = 2 ** i
    alpha_s = (s / 2) ** 0.5
    melspec = MelSpectrogram(sample_rate=sr, n_fft=s, hop_length=s//4, n_mels=8, ...).to(device)
    S_x   = melspec(x)
    S_G_x = melspec(G_x)
    loss  = (S_x - S_G_x).abs().sum() \
          + alpha_s * (((torch.log(S_x.abs()+eps) - torch.log(S_G_x.abs()+eps)) ** 2)
                        .sum(dim=-2) ** 0.5).sum()
    L += loss
return L
```

> - 这是 HiFi-GAN 那一套"多尺度 Mel 损失 + spectral convergence"，等价于一个 STFT-domain 的重建项。
> - 与 GAN 损失解耦（不依赖判别器），是 SoundStream 的核心重建信号。

---

### 4.4 判别器对抗损失  $\mathcal{L}_{\text{adv}}^{D}$（hinge 形式）

$$
\mathcal{L}_{\text{adv}}^{D}
=
\underbrace{\mathbb{E}\!\left[\operatorname{relu}\!\left(1 - D^{S}(s_x)\right)\right]
            + \sum_k \mathbb{E}\!\left[\operatorname{relu}\!\left(1 - D^{W^{(k)}}(x)\right)\right]}_{\text{real term}}
\;+\;
\underbrace{\mathbb{E}\!\left[\operatorname{relu}\!\left(1 + D^{S}(s_{G_x})\right)\right]
            + \sum_k \mathbb{E}\!\left[\operatorname{relu}\!\left(1 + D^{W^{(k)}}(G_x)\right)\right]}_{\text{fake term}}
$$

代码（`adversarial_d_loss`）：

```python
real_stft_loss      = F.relu(1 - features_stft_disc_x[-1]).sum(dim=3).squeeze()     / lengths_stft[-1].squeeze()
generated_stft_loss = F.relu(1 + features_stft_disc_G_x[-1]).sum(dim=-1).squeeze()  / lengths_stft[-1].squeeze()
real_wave_loss      = torch.stack([
    F.relu(1 - features_wave_disc_x[k][-1]).sum(dim=-1).squeeze() / lengths_wave[k][-1].squeeze()
    for k in wave_disc_names
], dim=1)
generated_wave_loss = torch.stack([
    F.relu(1 + features_wave_disc_G_x[k][-1]).sum(dim=-1).squeeze() / lengths_wave[k][-1].squeeze()
    for k in wave_disc_names
], dim=1)
real_loss      = torch.cat([real_stft_loss.view(-1,1), real_wave_loss],      dim=1).mean()
generated_loss = torch.cat([generated_stft_loss.view(-1,1), generated_wave_loss], dim=1).mean()
return real_loss + generated_loss
```

> 训练 G 时，`s_G_x` 与 `G_x` 通过计算图参与回传；训练 D 时使用 `.detach()` 截断梯度
> （见 `main.py:142, 187, 232`），保证 G 的更新不会通过 D 反向传播。

---

### 4.5 总损失（生成器）

$$
\boxed{\;
\mathcal{L}_G \;=\;
\underbrace{\lambda_{\text{adv}}}_{\text{1}}\cdot\mathcal{L}_{\text{adv}}^{G}
\;+\;
\underbrace{\lambda_{\text{feat}}}_{\text{100}}\cdot\mathcal{L}_{\text{feat}}
\;+\;
\underbrace{\lambda_{\text{rec}}}_{\text{1}}\cdot\mathcal{L}_{\text{rec}}
\;}
$$

代码（`criterion_g`）：

```python
criterion_g = lambda x, G_x, fS_x, fW_x, fS_G, fW_G, lW, lS: (
      LAMBDA_ADV * adversarial_g_loss(fS_G, fW_G, lS, lW)
    + LAMBDA_FEAT * feature_loss(fS_x, fW_x, fS_G, fW_G, lW, lS)
    + LAMBDA_REC  * spectral_reconstruction_loss(x, G_x)
)
criterion_d = adversarial_d_loss
```

> 注意权重差异：$\lambda_{\text{feat}}=100$ 远大于另两项 —— feature matching 在 SoundStream 中
> 是主要驱动 generator 学到判别器"眼中"真实波形结构的信号。

---

## 5. 优化器 & 训练循环要点（与 main.py 对齐）

```
optimizer_g = Adam(SoundStream.parameters(),       lr=1e-4, betas=(0.5, 0.9))
optimizer_d = Adam(wave_disc.parameters()
                 + stft_disc.parameters(),         lr=1e-4, betas=(0.5, 0.9))
```

每个 batch 内的更新顺序（对应 `main.py:104-150`）：

```
1. G_x  = soundstream(x)
2. s_x   = STFT(x)        ;   s_G_x = STFT(G_x)
3. fS_x = stft_disc(s_x)  ;   fW_x  = wave_disc(x)
4. fS_G = stft_disc(s_G_x);   fW_G  = wave_disc(G_x)
5. loss_g = criterion_g(x, G_x, fS_x, fW_x, fS_G, fW_G, lengths_wave, lengths_stft)
6. optimizer_g.zero_grad();  loss_g.backward();  optimizer_g.step()      # 训 G
7. 重新算 fS_x, fW_x；用 .detach() 算 fS_G_det, fW_G_det
8. loss_d = criterion_d(fS_x, fW_x, fS_G_det, fW_G_det, lengths_stft, lengths_wave)
9. optimizer_d.zero_grad();  loss_d.backward();  optimizer_d.step()      # 训 D
```

> 每个 step 内**先 G 后 D**；D 的 forward 用 `detach()` 阻止梯度流向 G。
> `lengths_*` 通过 `wave_disc.features_lengths(lengths_x)` 与 `stft_disc.features_lengths(lengths_s_x)`
> 提前算好，用于把 feature map 上的 sum 归一化到"单位有效样本"。

---

## 6. 总结 —— 四个 loss 在框图里的归属

```
                                     ┌──────────────────────────┐
                                     │                          │
                                     │     SoundStream (G)      │◀───────┐
                                     │                          │        │
                                     └────────────┬─────────────┘        │
                                                  │ G_x                  │ optimizer_g
                                                  ▼                      │
                ┌──────────────────┐   ┌──────────────────────────┐       │
   x ───────────▶│  STFT (手工算)  │──▶│       stft_disc (S)      │──┐    │
                └──────────────────┘   └──────────────────────────┘  │    │
                                                                   │    │
                                                                   ▼    │
                ┌──────────────────┐   ┌──────────────────────────┐  │    │
   x ───────────▶│  (直接进 wave)  │──▶│      wave_disc (W)       │──┤    │
                └──────────────────┘   └──────────────────────────┘  │    │
                                                                   │    │
        真实样本特征                                                 │    │
   ──────────────────────────────────────────────────────────────── │ ───┤
        生成样本特征（来自 G_x）                                     │    │
                                                                   ▼    ▼
                                                    ┌────────────────────────────┐
                                                    │  L_adv^G ── hinge on logits│──▶ optimizer_g
                                                    │  L_feat ── L1 on features  │──▶ optimizer_g
                                                    │  L_rec  ── 多尺度 mel+L1   │──▶ optimizer_g (only)
                                                    │  L_adv^D ── hinge(real,fake)│──▶ optimizer_d (only)
                                                    └────────────────────────────┘
```

| Loss | 依赖判别器？ | 输入 | 用于更新谁 |
|---|---|---|---|
| $\mathcal{L}_{\text{adv}}^G$ | ✅（最后一层 logits） | $D^S(s_{G_x}), D^{W^{(k)}}(G_x)$ | $G$ |
| $\mathcal{L}_{\text{feat}}$ | ✅（中间层 features） | 所有 feature maps | $G$ |
| $\mathcal{L}_{\text{rec}}$ | ❌（只用 $x, G_x$） | 多尺度 Mel 谱 | $G$ |
| $\mathcal{L}_{\text{adv}}^D$ | ✅（最后一层 logits） | 真实 & 生成的 logits（detach） | $D$ |

读到这里，再回去看 `main.py` 的 4 个损失函数 + 训练循环，应该就不会觉得"很乱"了 —— 它们正好对应这张图里的 4 条边。
