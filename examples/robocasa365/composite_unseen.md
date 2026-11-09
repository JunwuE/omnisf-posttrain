# Composite Unseen：在 Seen 完成后顺序训练

启动脚本：`scripts/train_robocasa365_composite_unseen_noalign_8gpu.sh`。
与 `train_robocasa365_composite_seen_noalign_8gpu.sh` 的训练参数相同：
8 卡、120000 步、全局 batch 512、梯度累积 1、FSDP 8、workers 32、关闭 alignment。
学习率、checkpoint 周期、日志周期也相同，使用原始 OmniSF 35999 权重初始化。
数据集为 `robocasa365-target-composite-unseen`，使用单独计算的全量归一化统计。

路径配置：`examples/robocasa365/composite_unseen.env.sh`。
默认输出目录：
`/share-mixed/project/robot-rfm/robot-exp/usr/ejw/posttrain/outputs/omnisf_composite_unseen`。
Seen 的新输出目录同样位于 `/share-mixed` 挂载下的 `omnisf_composite_seen`。
原 `/share/project` 输出保留；切换挂载是因为原存储在本次启动期间返回过 ENOSPC。

手动使用方式（在仓库根目录执行）：

```bash
bash scripts/train_robocasa365_composite_unseen_noalign_8gpu.sh --dry-run
bash scripts/train_robocasa365_composite_unseen_noalign_8gpu.sh --check-only
# 尚无本子集统计文件时运行；已有文件不会被覆盖
bash scripts/train_robocasa365_composite_unseen_noalign_8gpu.sh --prepare-stats
# 正式训练需要 8 张 GPU 全部空闲
bash scripts/train_robocasa365_composite_unseen_noalign_8gpu.sh
```

`--check-only` 可在 Seen 运行时检查 Unseen 的数据、权重、统计和 GPU 可见性，
不分配 GPU 显存；真正启动仍会拒绝与已有 GPU 计算进程重叠。
两个启动脚本共用 `launch_robocasa365_composite_seen.py` 的校验和日志逻辑，
以及 `compute_robocasa365_composite_seen_norm_stats.py --config-name ...` 的统计逻辑。

排队程序 `scripts/queue_robocasa365_composite_unseen.py` 使用独立的 tmux 会话
`robocasa_composite_unseen_queue`，每 60 秒检查指定 Seen 运行：

1. Seen 正常退出，状态为 `completed` 且退出码为 0。
2. Seen 的最终 checkpoint `119999` 包含参数、训练状态及已提交的元数据。
3. 8 张 GPU 已释放。
4. 执行 Unseen 启动脚本，启动时再次校验全部输入。

Seen 失败、进程消失或最终 checkpoint 不完整时，队列记录 `blocked` 并退出，
避免将失败误判成完成。队列有进程锁和一次性启动标记，防止重复启动。
该队列绑定具体的 Seen 运行；若手动换一个 Seen 运行，需更新并重新验证队列配置。

当前队列路径见输出目录的 `current_queue.json`，队列目录包含 `queue.json`、
`status.json` 和 `bootstrap.log`。Unseen 训练启动后，以其运行目录中的
`status.json` 为准；`train.log` 和 `metrics.csv` 会持续保存。
CSV 每 10 步记录训练器打印的平均 loss，保留 4 位小数，具体语义见
`composite_seen.md`。
