# RoboCasa365 后训练

RoboCasa365 已注册到 `openpi.training.config`，直接复用主仓库的
`scripts/train.py`、`scripts/train_align.py`、`scripts/compute_norm_stats.py` 和 `scripts/serve_policy.py`。
π0.5 使用 JAX 训练；可选择纯动作后训练，或使用 VGGT / OmniVGGT 的 PyTorch 特征提取分支进行空间对齐。
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

使用已准备好的 OmniSF JAX checkpoint；启用空间对齐时还需要 OmniVGGT 组件权重。
设置实际的预训练权重路径：

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

### 5.1 带 OmniVGGT 空间对齐

设置前面相同的数据、norm、JAX 权重、OmniVGGT 权重和 checkpoint 路径后运行：

```bash
# 当可见 GPU 多于 4 张时，指定本次使用的四张卡。
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/train_robocasa365_4gpu.sh
```

脚本复用 `train_align.py`，运行 60,000 个优化器更新。
全局 batch 256，每卡 64 个样本，梯度累积 1。
VGGT 特征提取在同四张卡上分块运行。
启动前检查数据、norm、权重和可见 GPU 数量。
`bash scripts/train_robocasa365_4gpu.sh --dry-run` 可预览命令；
恢复同一个实验使用 `bash scripts/train_robocasa365_4gpu.sh --resume`。

### 5.2 纯动作后训练，不使用 VGGT

`scripts/train_robocasa365_noalign_4gpu.sh` 直接调用 `scripts/train.py`，
复用 `robocasa365_target_atomic` 的数据、norm、π0.5 模型和优化器设置。
它显式关闭 `align_enabled`、将 `align_loss_coeff` 设为 0，并清空 VGGT 权重路径；
训练仅计算动作 flow-matching loss，不创建 VGGT、对齐投影头或特征提取 worker。
不需要设置 `ROBOCASA365_VGGT_PATH`。

先设置 `ROBOCASA365_DATA_ROOT`、`ROBOCASA365_ASSETS_DIR`、
`ROBOCASA365_PRETRAIN_PARAMS` 和 `ROBOCASA365_CHECKPOINT_DIR`，
或 source 已有的路径配置文件，再设置独立实验名：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export EXP_NAME="robocasa365_atomic_noalign_4gpu_$(date +%Y%m%d_%H%M%S)"
bash scripts/train_robocasa365_noalign_4gpu.sh
```

脚本自动激活 `omnisf` Conda 环境，默认 60,000 步、全局 batch 256、
梯度累积 1、FSDP 4、数据 worker 16，开启本地快速加载；
每 10 步打印日志，每 5,000 步保存 checkpoint。
JAX 默认使用可复用显存分配的 `default` allocator，并关闭启动时预分配；
可通过 `XLA_PYTHON_CLIENT_ALLOCATOR` 环境变量覆盖。
末尾追加训练参数可覆盖默认值，例如 `--log-interval 100`。

`--dry-run` 仅打印启动命令，`--check-only` 检查数据、norm、JAX 权重路径和四卡可见性，
两者都不会启动训练。检查卡数不代表显卡空闲，应等原训练结束或使用另外四张空闲卡。
续训时保持同一个纯动作实验的 `EXP_NAME`，追加 `--resume`；
带对齐训练的 checkpoint 含有额外模型和优化器状态，不能直接作为本入口的续训 checkpoint。
已有 OmniSF 的 `params` 仍可通过 `ROBOCASA365_PRETRAIN_PARAMS` 用作初始化权重，
权重加载器仅选取 π0.5 模型对应的参数。

日志中的 `train_step_s` 为 JAX 更新耗时，`data_wait_s` 为取下一批数据的耗时。
移除在线 VGGT 后预计耗时会下降，实际速度需在编译结束、训练稳定后测量。

### 5.3 三个 target 子集：单机 8 卡、batch 512、120,000 步，无 align

入口：`scripts/train_robocasa365_mixed_noalign_8gpu.sh`。
复用 `robocasa365_target_mixed` 和 `scripts/train.py`，一个 Python 进程使用 8 张 GPU，
全局 batch 512（每卡 64）、梯度累积 1、FSDP 8，只计算动作 loss。
不需要 OmniVGGT 权重，也不通过 `torchrun` 启动多个进程。

默认 warmup 1,000 步、峰值学习率 `2.5e-5`，cosine 衰减到第 120,000 步的 `2.5e-6`。
`--num-workers 32` 在现有 mixed 加载器中表示 32 个读取线程，共享一份数据集；
可追加 `--num-workers 16` 调低。每 10 步记录日志，每 5,000 步保存，
保留最新 checkpoint 及每 20,000 步的 checkpoint，便于中途评估；追加 `--keep-period None`
可改成只留最新一份。

启动脚本自动加载 `examples/robocasa365/mixed_noalign_8gpu.env.sh` 的共享存储路径，
新终端无需手动 export 四个必需路径；显式设置的环境变量优先。
`OPENPI_DATA_HOME` 默认指向 `/share/project/robot-rfm/cache/openpi`，复用已经准备好的
`big_vision/paligemma_tokenizer.model`。新服务器不需要再次从 Google 存储下载分词器；
启动检查会验证文件存在且能被 SentencePiece 加载。显式设置 `OPENPI_DATA_HOME` 时，
该目录也必须包含相同的分词器文件。若旧进程已经停在下载处，终止后重新启动才能使用新配置。
DATA_ROOT 默认与已有 atomic 相同，包含三个独立的 **v2.1** 子集。
训练时直接从这三个目录在线采样，不创建额外的 mixed 数据集。
路径默认值如下，手动准备数据时也可直接 source 该文件：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate omnisf
cd /home/junwue/code/omnisf
export ROBOCASA365_SOURCE_DATA_ROOT=/share-mixed/project/robot-rfm/robot-raw-data/robocasa365
export ROBOCASA365_DATA_ROOT=/share-mixed/project/robot-rfm/robot-raw-data/robocasa365_lerobot2
export ROBOCASA365_ASSETS_DIR=/share/project/robot-rfm/robot-exp/usr/ejw/posttrain/outputs/omnisf_mixed/assets
export ROBOCASA365_CHECKPOINT_DIR=/share/project/robot-rfm/robot-exp/usr/ejw/posttrain/outputs/omnisf_mixed/checkpoints
export ROBOCASA365_PRETRAIN_PARAMS=/share/project/robot-model/omnisf/35999/omnisf_9.0_jax_35999/params
export ROBOCASA365_MIXED_WEIGHTS=1,1,1
export EXP_NAME=robocasa365_target_mixed_noalign_120k_bs512_8gpu
```

若两个 composite 子集尚未转换，可将它们分别转换到 atomic 旁边；
三个子集都已经准备好时，跳过转换。下面是可重用的准备命令：

```bash
(
    set -e  # 转换失败时停止这段准备流程。
    mkdir -p "$ROBOCASA365_DATA_ROOT"
    for split in composite-seen composite-unseen; do
        if [[ ! -f "$ROBOCASA365_DATA_ROOT/robocasa365-target-$split/meta/info.json" ]]; then
            python examples/robocasa365/convert_lerobot_v3_to_v21.py \
                --source "$ROBOCASA365_SOURCE_DATA_ROOT/robocasa365-target-$split" \
                --output "$ROBOCASA365_DATA_ROOT/robocasa365-target-$split" --workers 8
        fi
    done

)
```

`mixed` 归一化统计与混合采样是两个独立步骤。统计文件只是 state/action 的均值、标准差和
分位数，用于训练输入与推理输出的缩放，不合并或复制数据集。
已有来源、采样权重兼容的统计文件时可直接使用；需要新统计时，在 CPU 上运行：

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu python scripts/compute_norm_stats.py \
    --config-name robocasa365_target_mixed --batch-size 512 --num-workers 4
```

默认 `1,1,1` 表示按各子集帧数混合，当前三组采样概率约为 **14.9%、40.1%、45.0%**。
若希望三组各占约 1/3，可将权重设成各组 `meta/info.json` 中 `total_frames` 的倒数；
改变混合比例后需在独立 assets 目录重新计算统计，并使用新实验名。
启动检查会验证统计维度和有限值，但不能证明外部统计文件的数据来源或权重，
因此不能直接沿用 atomic 的统计或来源不明的 mixed 统计。

在具有 8 张可用 GPU 的节点执行：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash scripts/train_robocasa365_mixed_noalign_8gpu.sh --dry-run
bash scripts/train_robocasa365_mixed_noalign_8gpu.sh --check-only
bash scripts/train_robocasa365_mixed_noalign_8gpu.sh
# 继续本实验时，使用相同的环境与 EXP_NAME：
# bash scripts/train_robocasa365_mixed_noalign_8gpu.sh --resume
```

`--check-only` 校验三组数据的文件清单、mixed norm、本地分词器、预训练权重、可见卡数和实验目录，
不会创建训练进程；文件清单检查不代替视频内容解码验证，也不检查 GPU 是否空闲。
`--resume` 恢复已有的完整 checkpoint。如果启动阶段中断、实验目录仍为空，
也可使用 `--resume` 重试：训练器会从预训练参数初始化，从第 0 步开始，保留该目录。
目录含有不完整 checkpoint 或其他文件、却没有可恢复状态时仍会报错。
带 align 的 atomic checkpoint 不能作为这里的 `--resume` 训练状态；
新实验从 `ROBOCASA365_PRETRAIN_PARAMS` 初始化 π0.5。

120,000 × 512 = 61,440,000 次帧起点采样，按目前约 1,496 万帧计算约为 4.1 轮等效采样，
并非保证每帧恰好遍历四次。建议用保留的中间 checkpoint 分别评估三个子集，
根据成功率判断是否需要完整跑到 120,000 步。
这是三个 target 子集的联合后训练；[官方 foundation-model 设置](https://robocasa.ai/docs/build/html/benchmarking/foundation_model_learning.html)
是三个子集分别微调，报告结果时应注明训练数据范围。

### 5.4 单独训练 composite_seen / composite_unseen，并保存 loss

两个子集均沿用 mixed 的 8 卡、120,000 步、全局 batch 512、关闭 alignment 配方，
分别从原始 OmniSF 35999 权重初始化，使用各自的全量归一化统计和独立输出目录。

- [Composite Seen 启动、统计及日志说明](composite_seen.md)
- [Composite Unseen 启动及顺序排队说明](composite_unseen.md)

启动脚本自动保存完整 `train.log` 和每 10 步打印的区间平均 loss `metrics.csv`。
Seen → Unseen 队列会等待 Seen 正常结束、最终 checkpoint 保存完整及 GPU 释放，
随后启动 Unseen。相关路径可通过各自的 `.env.sh` 或环境变量设置。

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
