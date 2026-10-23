# RoboCasa365 后训练

RoboCasa365 已注册到 `openpi.training.config`，直接复用主仓库的
`scripts/train_align.py`、`scripts/compute_norm_stats.py` 和 `scripts/serve_policy.py`。
π0.5 使用 JAX 训练，VGGT / OmniVGGT 使用现有 PyTorch 特征提取分支进行空间对齐。
运行时不需要 `omnisf_robocasa365`、动态注册脚本或修改 `PYTHONPATH`。

## 1. 环境与数据

使用已配置的 Conda 环境，从本仓库根目录执行：

```bash
conda activate omnisf
cd /home/junwue/code/omnisf
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

当前固定的 LeRobot 版本读取 **v2.1**。如果数据来自 v3.0 镜像，需要先完成
v3.0 → v2.1 转换；本次接入复用现有 v2.1 加载器，没有引入新版 LeRobot。
直接在 `omnisf` Conda 环境中转换，保持原始数据与输出分开存放：

```bash
python examples/robocasa365/convert_lerobot_v3_to_v21.py \
  --source /path/to/robocasa365/robocasa365-target-atomic \
  --output /path/to/robocasa365_lerobot2/robocasa365-target-atomic \
  --workers 8
```

转换器保留状态、动作与任务信息，按 episode 无损拆分视频，并将时间戳归零。
会逐文件校验 Parquet 回读结果、逐视频校验帧数和首帧时间戳；
要求源 Parquet 文件按 episode 分界、视频 episode 边界为关键帧，否则明确报错。
未完成结果保存在 `.incomplete` 目录，同一命令可断点继续；不会覆盖已完成的输出。
`meta/conversion.json` 记录来源和校验范围。LeRobot 元数据统计只包含数值字段，
没有编造源数据缺失的图像统计；模型使用下面独立计算的 state/actions norm。

指定存放三个子集的父目录，训练和计算统计时使用同一个设置：

```bash
export ROBOCASA365_DATA_ROOT=/path/to/robocasa365_v21
```

默认目录是本仓库的 `data/robocasa365_v21`，结构如下：

```text
robocasa365_v21/
  robocasa365-target-atomic/
    meta/{info.json,tasks.jsonl,episodes.jsonl,episodes_stats.jsonl}
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/<camera_key>/episode_000000.mp4
  robocasa365-target-composite-seen/
    ...
  robocasa365-target-composite-unseen/
    ...
```

| 配置名 | 数据 |
| --- | --- |
| `robocasa365_target_atomic` | Atomic |
| `robocasa365_target_composite_seen` | Composite Seen |
| `robocasa365_target_composite_unseen` | Composite Unseen |
| `robocasa365_target_mixed` | 上述三组混合，不需要建立额外的 mixed 数据目录 |

数据映射保留原始控制语义：

| LeRobot 字段 | 模型输入 |
| --- | --- |
| `observation.state` | 原始 16 维状态，补零到 32 维 |
| `action` | 原始 12 维控制，补零到 32 维 |
| `observation.images.robot0_agentview_left` | `base_0_rgb` |
| `observation.images.robot0_eye_in_hand` | `left_wrist_0_rgb` |
| `observation.images.robot0_agentview_right` | `right_wrist_0_rgb` |
| `task_index` + `meta/tasks.jsonl` | 任务文本 |

12 维动作包含底盘运动 4 维、控制模式 1 维、末端位移 3 维、旋转 3 维、夹爪 1 维。
训练与推理均保持这些数值，不做额外 delta 变换、单位缩放或夹爪反转。
这与原有 `LeRobotRoboCasaNoMovementDataConfig` 的 7 维动作接口不同。

## 2. 归一化统计

```bash
python scripts/compute_norm_stats.py \
  --config-name robocasa365_target_atomic --batch-size 256 --num-workers 4
```

切换配置名即可计算其他子集，统计写入
`assets/<config_name>/norm_stats.json`。沿用主仓库的统计流程，读取状态和动作，
不解码视频；模型输入仍使用分位数归一化。首次构造数据变换可能下载 PaliGemma tokenizer。

混合训练共用一份混合统计，不能拿单个子集的统计代替：

```bash
export ROBOCASA365_MIXED_WEIGHTS=1,1,1
python scripts/compute_norm_stats.py \
  --config-name robocasa365_target_mixed --batch-size 256 --num-workers 4
```

顺序为 atomic、composite_seen、composite_unseen。
采样概率正比于 **权重 × 子集帧数**，所以 `1,1,1` 并非三组等概率。
修改为 `2,1,1` 后，应重新计算 mixed 统计，再使用相同设置训练。
已有参考项目的统计也可放入上述目录，但须确认数据、动作语义和混合权重一致。

## 3. 使用现有空间对齐训练入口

先设置真实的预训练权重路径：

```bash
# JAX checkpoint 的 params 目录，不是 PyTorch safetensors。
export ROBOCASA365_PRETRAIN_PARAMS=/path/to/omnisf_checkpoint/params
# 目录内应包含 OmniVGGT.safetensors。
export ROBOCASA365_VGGT_PATH=/path/to/omnivggt
```

默认预训练位置为 `checkpoints/omnisf_9.0_jax/experiments/35999/params`，
默认对齐权重位置为 `checkpoints/omnivggt`。也可将预训练路径设为本地 `pi05_base/params`。
OmniVGGT 构造器还会查找 `checkpoints/dinov2/dinov2_vitl14_reg4_pretrain.pth`，
不存在时会走原有的 `torch.hub` 下载流程。

单卡启动：

```bash
python scripts/train_align.py robocasa365_target_atomic \
  --exp-name robocasa365_atomic \
  --align-target-model omnivggt \
  --no-wandb-enabled
```

默认按单卡设置：micro batch = 1、梯度累积 = 2、FSDP = 1、VGGT 分块大小 = 1。
实际显存需求仍取决于权重和 JAX 编译，完整模型训练需要实测。
其余算法设置沿用参考实现：20,000 个优化器更新、EMA 0.999、状态丢弃概率 0.25、
VLA 第 12 层对齐 VGGT 最后一层、对齐损失权重 0.5、三路相机、无相机参数。
学习率使用 1,000 步 warmup、峰值 2.5e-5、100,000 步余弦周期及下限 2.5e-6；
20,000 步训练不会走完整个衰减周期。

复现参考实现的 8 卡吞吐设置时，使用已有的单进程 JAX 多卡入口：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python scripts/train_align.py robocasa365_target_atomic \
  --exp-name robocasa365_atomic_8gpu \
  --align-target-model omnivggt \
  --batch-size 512 --gradient-accumulation-steps 2 \
  --fsdp-devices 8 --align-vggt-devices 8 \
  --align-feature-batch-size 32 --num-workers 8
```

这里有效 batch = 1024；它与默认单卡配置的优化行为不同。
若使用原版 VGGT，设置 `--align-target-model vggt --vggt-weight-path /path/to/vggt`，
对应目录内需要 `model.pt`。

混合训练仅替换配置名：

```bash
python scripts/train_align.py robocasa365_target_mixed \
  --exp-name robocasa365_mixed --align-target-model omnivggt --no-wandb-enabled
```

默认仅保存最后一次 checkpoint（步号 `19999`），避免频繁保存完整模型和优化器占用空间。
需要中途保存时加 `--save-interval 5000`；继续同一实验加 `--resume`。
Checkpoint 和统计目录分别可通过 `ROBOCASA365_CHECKPOINT_DIR`、`ROBOCASA365_ASSETS_DIR` 修改。
训练参数使用主仓库 CLI 覆盖，不再依赖参考项目中用于设置 batch、学习率等的 shell 变量。

默认执行 LeRobot 的文件及时间戳校验。确认转换数据完整后，可加
`--data.fast-local-loading`，单集和 mixed 都支持；快速模式仍检查数据格式版本。

## 4. Atomic：60,000 步、全局 batch 256、8 卡

先在服务器设置数据和权重位置，并计算 atomic 归一化统计（已有匹配的统计可直接使用）：

```bash
conda activate omnisf
cd /path/to/omnisf-posttrain
export ROBOCASA365_DATA_ROOT=/path/to/robocasa365_v21
export ROBOCASA365_PRETRAIN_PARAMS=/path/to/omnisf_checkpoint/params
export ROBOCASA365_VGGT_PATH=/path/to/omnivggt
export ROBOCASA365_ASSETS_DIR="$PWD/assets"
export ROBOCASA365_CHECKPOINT_DIR=/path/to/training_checkpoints

python scripts/compute_norm_stats.py \
  --config-name robocasa365_target_atomic --batch-size 256 --num-workers 8
```

在已经分配到 8 张 GPU 的机器上直接启动：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/train_align.py robocasa365_target_atomic \
  --exp-name robocasa365_target_atomic_60k_bs256 \
  --num-train-steps 60000 \
  --batch-size 256 --gradient-accumulation-steps 1 \
  --fsdp-devices 8 \
  --align-target-model omnivggt --align-device-mode shared \
  --align-vggt-devices 8 --align-feature-batch-size 16 \
  --num-workers 16 \
  --lr-schedule.decay-steps 60000 \
  --save-interval 5000 --keep-period None \
  --no-wandb-enabled
```

这里 `256` 是 8 卡合计的 batch，每卡 32；累积设为 1 后，有效 batch 也是 256。
`60000` 是优化器更新次数。学习率衰减周期同步设为 60,000 步，warmup 和峰值沿用配置；
若要保持原始 100,000 步衰减周期，可改为 `--lr-schedule.decay-steps 100000`。
每 5,000 步保存一次，`keep-period None` 配合现有 checkpoint 管理器只保留最新一份，
最后一次保存的目录步号为 `59999`。

Slurm 服务器使用 [8 卡 sbatch 脚本](../../slurm_scripts/train_robocasa365_8gpu.sbatch)：

```bash
# 在仓库根目录提交；上面的数据、权重和输出目录环境变量会传入作业。
export CONDA_BASE="$(conda info --base)"
export CONDA_ENV=omnisf
export PROJECT_ROOT="$PWD"
export CONFIG=robocasa365_target_atomic
export EXP_NAME=robocasa365_target_atomic_60k_bs256

# 先预览实际命令，不申请 GPU，也不启动训练。
bash slurm_scripts/train_robocasa365_8gpu.sbatch --dry-run

# 替换为服务器实际的分区；若集群要求账户，再加 --account=你的账户。
sbatch --partition=你的GPU分区 slurm_scripts/train_robocasa365_8gpu.sbatch
```

脚本申请 1 个节点、8 张 GPU、1 个任务、32 个 CPU、256 GB 内存和 48 小时运行时限。
其中 48 小时只是作业时限，不代表预计训练耗时，可通过 `sbatch --time=... --mem=...` 覆盖。
JAX 使用单个进程管理 8 卡；脚本沿用 Slurm 设置的 `CUDA_VISIBLE_DEVICES`。
分区和账户按集群实际配置传入，原有 HKUST 脚本中的 `acd_u` 不作为通用默认值。
Slurm 获得资源后会自动执行训练；日志位于提交目录的
`omnisf-robocasa365-<job_id>.out` / `.err`。
参见 [Slurm sbatch 文档](https://slurm.schedmd.com/sbatch.html)。

查看任务与续训：

```bash
squeue -u "$USER"
tail -f omnisf-robocasa365-<job_id>.out

# 保持 CONFIG、EXP_NAME 和 checkpoint 目录相同；总目标仍为 60,000 步。
sbatch --partition=你的GPU分区 slurm_scripts/train_robocasa365_8gpu.sbatch --resume
```

若训练显存不足，可用 `--batch-size 128 --gradient-accumulation-steps 2`，
保持有效 batch 为 256；若是 VGGT 特征提取显存不足，可减小 `--align-feature-batch-size`。
这些参数可追加在 sbatch 脚本名后覆盖默认值。切换数据子集时设置 `CONFIG` 并准备对应统计。

## 5. Atomic：本机 4 卡，不使用 Slurm

设置前面相同的数据、norm、JAX 权重、OmniVGGT 权重和 checkpoint 路径后运行：

```bash
# 当可见 GPU 多于 4 张时，指定本次使用的四张卡。
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/train_robocasa365_4gpu.sh
```

脚本复用 `train_align.py`，运行 60,000 个优化器更新。
沿用 8 卡方案每卡 32 个样本的设置，4 卡采用 micro batch 128、梯度累积 2，
有效全局 batch 仍为 256。VGGT 特征提取在同四张卡上分块运行。
启动前检查数据、norm、权重和可见 GPU 数量。
`bash scripts/train_robocasa365_4gpu.sh --dry-run` 可预览命令；
恢复同一个实验使用 `bash scripts/train_robocasa365_4gpu.sh --resume`。

## 6. 推理

```bash
python scripts/serve_policy.py policy:checkpoint \
  --policy.config robocasa365_target_atomic \
  --policy.dir checkpoints/robocasa365_target_atomic/robocasa365_atomic/19999
```

客户端发送 `observation/state`、`observation/image`、`observation/wrist_image`、
`observation/right_image` 和 `prompt`；相机顺序对应上表。
图像支持 uint8 HWC 或 LeRobot 风格的 float CHW [0, 1]。
服务端沿用 checkpoint 内保存的统计，输出 `(action_horizon, 12)` 原始控制动作。

## 7. 回归测试

```bash
JAX_PLATFORMS=cpu python -m pytest -q \
  src/openpi/policies/robocasa365_policy_test.py \
  src/openpi/training/robocasa365_test.py
```

测试生成小型 LeRobot v2.1 Parquet 和 H.264 视频，覆盖完整动作、三相机、
单集／混合统计、普通／快速读取、JAX batch、归一化逆变换及版本不匹配。
测试中的 tokenizer 使用离线替身，不加载真实 π0.5 / VGGT 权重。
