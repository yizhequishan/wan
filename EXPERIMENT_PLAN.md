# 实验计划：视频 DiT 的实体—关系级机制可解释性

主假说：Wan2.1 视频 DiT 的物理计算不是均匀分布在全部 token 连接中，而是形成由实体内边、
轨迹边和实体间交互边组成的动态稀疏子图 G_t^(l,τ)。

三个待验证命题：**可读出性**（实体间注意力能预测接触/支撑/遮挡/碰撞事件）、
**增量信息**（超出几何变量 d_ij、Δv_ij 之外仍有信息）、**因果作用**（干预实体间边导致
接触失败、穿透、碰后方向错误）。

工作方式：Windows 本地（D:\keyan\wan）编辑代码 → git push → 远程 Linux 服务器（2×RTX 3090）
pull 并执行。所有实验脚本放 `experiments/`，输出统一写 `results/<exp_name>/`，长任务用
tmux + nohup，日志用 wandb（离线模式亦可）。

---

## 硬性约束（写在最前面，全程遵守）

1. **永不物化完整注意力矩阵。** 480×832×81 帧 → 32,760 token，单层单头 fp16 逻辑矩阵
   ≈ 2.1 GB。只计算并存储实体块聚合量 E_ij（n_ent × n_ent，逐层逐头逐步）。
2. **第一阶段实体节点来自合成数据真值 mask**，不用 SAM2，不用 k-means 簇。实体定义的噪声
   必须与"模型没有交互表征"这一失败模式严格分离。
3. **3090 是 sm86：只有 FlashAttention-2，没有 FA3。** 需要 logits 的层改走手写分块
   attention（repo 已有 SDPA fallback，`wan/modules/attention.py:164`）。
4. 双卡并行方式是**按视频切分数据并行**（两个进程各占一卡），不做模型并行。
5. 每个实验脚本可从命令行完整复现：`python experiments/xx.py --config configs/xx.yaml`。

---

## Phase 0：环境与冒烟测试（第 1 周，远程服务器）

- [ ] conda 环境：Python 3.10 + PyTorch ≥ 2.4 (cu121) + `pip install -r requirements.txt`；
      flash-attn 2 从预编译 wheel 装（sm86 源码编译约 1 小时，能避则避）。
- [ ] 下载权重：`Wan2.1-T2V-1.3B`（DiT ≈ 2.5 GB + umt5-xxl 编码器 ≈ 11 GB + VAE）。
- [ ] 冒烟测试（单卡 24 GB 需 offload）：

```bash
python generate.py --task t2v-1.3B --size 832*480 \
  --ckpt_dir ./Wan2.1-T2V-1.3B --offload_model True --t5_cpu \
  --sample_shift 8 --sample_guide_scale 6 \
  --prompt "A red ball rolls and collides with a blue block on a wooden table."
```

- [ ] 记录：单卡峰值显存、50 步耗时（后续估算总机时用）。
- 备注：分析阶段文本编码可一次性预计算缓存（受控数据 prompt 固定），T5 之后不再加载，
  省 11 GB 内存和加载时间。

## Phase 1：受控数据集（第 1–2 周，与 Phase 0 并行）

用 Kubric（Docker，CPU 渲染，不占 GPU）自渲染，规格对齐 Wan：**832×480、81 帧、16 fps**。

三类场景，各 150–200 条，总计约 500 条；按 7:1.5:1.5 划分 train/val/test，**按场景划分**
（同一场景的不同视角/种子不得跨集）：

| 场景 | 事件真值 | 用途 |
|---|---|---|
| 双球/多体碰撞 | 接触时刻、碰前碰后速度 | 接触预测、碰撞干预 |
| 堆叠与支撑（含抽走支撑物） | 支撑关系对、坍塌时刻 | 支撑关系 probe |
| 遮挡穿行（物体经过遮挡物后重现） | 遮挡区间、身份对应 | 遮挡/身份持续性 |

每条视频保存：RGB（mp4）、逐帧实例 mask（png 序列）、逐帧物体状态
（3D 位置/速度/包围盒，json）、事件标注（接触对 + 帧号）。

**mask → token 对齐**（关键基建，单独写成模块 + 单测）：
- 空间：mask 下采样 8×（VAE）再 2×2 patchify → 每 16×16 像素块投票，≥50% 归属该实体；
- 时间：latent 帧 0 对应视频帧 0，latent 帧 t≥1 对应视频帧 [4t-3, 4t]，取中间帧 mask；
- 边界/多数票不过半的 token 归入 boundary 类，**从 E_ij 统计中剔除**（不算背景也不算实体）；
- 单测：随机场景上可视化 token 归属叠加图，人工抽查 20 条。

## Phase 2：Inversion + E_ij 提取管线（第 2–4 周）

### 2.1 Flow inversion（参照 Invisible Hand 的做法，它已在 WAN-1.3B 上验证可行）

对真实（渲染）视频：VAE 编码 → 沿学到的 velocity field **反向积分**（Euler，与
`wan/text2video.py:206` 的 unipc 正向采样共用 timestep 网格）→ 得到每个去噪步 τ 的
中间 latent。分析用 τ 网格取稀疏 10 步即可（τ ∈ {0.1, 0.2, ..., 1.0}），不需要 50 步全存。

验收标准：inversion 后再正向去噪，重建视频与原视频 PSNR > 25 dB（低于此说明积分步数
或 shift 参数有问题，先修再往下走）。

### 2.2 注意力 hook 与块级聚合

- Hook 点：`wan/modules/model.py:105` `WanSelfAttention.forward`（`model.py:149` 的
  flash_attention 调用旁加旁路）。给 forward 加可选参数 `entity_masks`，非 None 时额外
  走一条分块手写路径：
  - 对每对实体 (i,j)：取 Q 的 i-token 行、K 的 j-token 列，分块 einsum 得 logit 块，
    与全行 logsumexp（分块累积）组合出该块的 softmax 概率质量 E_ij = 平均注意力质量；
  - 逐头保存，fp32；显存峰值 O(chunk × L)，chunk 取 2048。
- 输出格式：每条视频一个 npz/HDF5：`E[τ=10, layer=30, head=12, n_ent, n_ent]` + 元数据。
  单条视频 < 10 MB，500 条 < 5 GB，无存储压力。
- **管线正确性验证（必做，先于一切 probe）**：
  1. 对角线检查：E_ii（实体内）应显著高于随机块；
  2. 复现 DiffTrack 式结论作为阳性对照：用相同 hook 提取 query-key 相似度，验证跨帧
     对应在特定层随去噪增强。若复现不出，管线有 bug，不是科学结论。

预算估算：单条视频 × 10 步 × 30 层 hook，在冒烟测试耗时基础上乘 ~3–5；500 条双卡
预计 2–4 天，可接受。若太慢：先只 hook 全部层跑 50 条做层定位，再对 top-5 层跑全量。

## Phase 3：Probe 与增量信息（第 4–6 周）→ **止损点 M1**

### 3.1 读出任务

对每对实体 (i,j)、每个 latent 帧，预测：
(a) 当前是否接触；(b) 未来 k∈{2,4,8} 帧内是否接触；(c) 是否为支撑关系；
(d) 是否处于遮挡；(e) 碰后分离方向（左/右二分类，用碰撞场景子集）。

### 3.2 特征与基线（四档，逐级对比）

| 档 | 特征 | 回答的问题 |
|---|---|---|
| B0 几何 | d_ij、Δv_ij、包围盒尺寸、接近率 | 视觉接近就够了吗 |
| B1 VAE | 实体区域 VAE latent 池化 | 信息是否在压缩表征里就有 |
| B2 光流 | RAFT 光流实体区域池化 | 任何运动特征都能预测吗 |
| Ours | B0 + E_ij 轨迹（选定层/头/步） | 增量 ΔAUC |

分类器统一用 logistic regression（主结果）+ 2 层 MLP（附录），按场景 5-fold。

### 3.3 定位分析

AUC 关于 (layer, head, τ) 的三维热图 → 找出"交互层/交互头"。同时做**成组聚合**版本
（全头拼接、top-k 头拼接），应对"分布式表征"情形。

### 3.4 止损判据（M1，第 6 周末执行，写进周报不许拖）

- **通过**：存在 (layer, head, τ) 或成组组合使 AUC(B0+E) − AUC(B0) ≥ 0.03，且对 B1、B2
  同样成立，且在 val/test 一致 → 进入 Phase 4。
- **失败**：所有组合 ΔAUC < 0.03 → 主假说不成立。转向撰写"分布式交互表征"报告
  （定位分析 + 阴性结果 + DiffTrack 阳性对照仍是一篇可投的分析论文），并重新评估方向。

## Phase 4：因果干预（第 6–10 周）

### 4.1 干预对象与操作

在**生成过程**（不是 inversion）中操作。为保证有事件真值：取受控视频，正向加噪到
中间步 τ₀（SDEdit 式 renoise），再去噪回来——base 轨迹已知会发生碰撞/支撑/遮挡。

干预：对选定层集合 S、实体对 (i,j)，logit 块 L_ij ← L_ij − λ 后重新 softmax
（在手写分块路径里实现，λ ∈ {2, 5, ∞}）。

### 4.2 对照组设计（缺一不可）

1. **匹配对照边**：相同 token 数、相似空间距离、相近注意力质量、但无真实交互的实体对；
2. **随机 token 组**：同 token 数的随机 token 集合；
3. **λ 剂量曲线**：效应应随 λ 单调；
4. **干预窗口敏感性**（应对去噪自修复，DiffTrack 已证明对应随去噪增强）：
   单步 τ、连续窗口 [τ₁,τ₂]、全程，三种都报告。自修复现象本身作为发现写入论文。

### 4.3 效应度量（自动化，基于生成结果的跟踪）

- 接触时刻误差（帧）；穿透率（实体 mask IoU > 阈值的帧占比，mask 用 SAM2 从首帧真值
  mask 传播——此处 SAM2 只用于**度量生成结果**，不用于定义节点，不污染机制结论）；
- 碰后方向误差（度）；遮挡后身份恢复失败率；对象数量变化。
- 每条件 ≥ 50 个种子，报告均值 ± bootstrap CI；判据：真实交互边干预效应显著强于
  两类对照（配对检验，p < 0.01 + 效应量）。

## Phase 5：应用闭环 + 论文（第 10–16 周）

- **方向二（同一篇论文的应用章节）**：预算分配器 b_ij = f(E_ij 因果层, P_contact, τ)。
  实现用 block mask + SDPA / FlexAttention（3090 上性能一般但足够出对比数字）。
  目标：**同 FLOPs 下物理指标 > uniform / TopK / SVG2**；> dense 只作 bonus，不押注。
- **方向四（同一篇论文一个 section）**：E_ij 时序拓扑统计（边持续性、突发、互惠性、熵）
  训练轻量验证器判物理合理性；跨模型测一次 CogVideoX-2B 泛化。
- 评测：Physics-IQ Verified + 自建受控指标（接触时刻误差、穿透率、轨迹突变、
  对象数量变化、遮挡后身份恢复、支撑保持、碰后方向）。不单报总分。

---

## 里程碑与止损总表

| 周 | 里程碑 | 止损/验收 |
|---|---|---|
| 1 | 环境 + 冒烟测试 | 单卡能生成 832×480×81 |
| 2 | Kubric 数据 v1（≥100 条）+ mask对齐单测 | 可视化抽查通过 |
| 4 | inversion（PSNR>25）+ E_ij 管线 + DiffTrack 阳性对照复现 | 复现失败→修管线，不往下走 |
| 6 | **M1：probe 结果** | ΔAUC≥0.03 →继续；否则转分布式报告 |
| 10 | **M2：干预结果** | 真实边 vs 对照边显著性 |
| 16 | M3：应用章节 + 初稿 | — |

## 目录结构约定

```
experiments/
  configs/            # yaml，每实验一份
  data/kubric/        # 渲染脚本 + 场景定义
  s00_smoke.py
  s01_align_masks.py  # mask→token 对齐 + 单测
  s02_invert.py       # flow inversion，验收 PSNR
  s03_extract_eij.py  # hook + 块级聚合，输出 HDF5
  s04_probe.py        # B0-B2 基线 + ours，输出 AUC 热图
  s05_intervene.py    # 干预 + 对照
  s06_metrics.py      # 穿透率/接触误差等自动度量
results/<exp_name>/   # 服务器上，git-ignore
```
