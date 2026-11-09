# Composite Seen：8 卡、120k、batch 512、无 alignment

启动脚本：`scripts/train_robocasa365_composite_seen_noalign_8gpu.sh`。
训练参数沿用 `train_robocasa365_mixed_noalign_8gpu.sh`，仅改为单独的
`robocasa365_target_composite_seen` 配置及独立的数据统计、输出目录。
初始化仍使用原始 OmniSF 35999 权重；不从 mixed 120k 权重继续训练。

- 单进程控制 8 张 GPU，FSDP 8；全局 batch 512，梯度累积 1。
- 120000 个 optimizer updates；学习率 warmup 1000，峰值 2.5e-5，
  cosine decay 至 2.5e-6；数据加载 workers 32。
- 关闭 alignment 和 VGGT，使用 `scripts/train.py` 的动作 loss。
- 每 5000 步保存 checkpoint，保留 20000 步倍数及最近 checkpoint；最终保存 119999。
- 路径在 `examples/robocasa365/composite_seen.env.sh`，可以通过环境变量覆盖。

在仓库根目录执行：

```bash
# 查看命令 / 检查完整数据文件、归一化统计、权重、tokenizer 和空闲 GPU
bash scripts/train_robocasa365_composite_seen_noalign_8gpu.sh --dry-run
bash scripts/train_robocasa365_composite_seen_noalign_8gpu.sh --check-only

# 首次部署且尚无本子集的统计文件时运行；已有统计时会拒绝覆盖
bash scripts/train_robocasa365_composite_seen_noalign_8gpu.sh --prepare-stats

# 正式训练；建议放在独立 tmux 会话
bash scripts/train_robocasa365_composite_seen_noalign_8gpu.sh

# 训练停止且已有完整 checkpoint 时恢复
bash scripts/train_robocasa365_composite_seen_noalign_8gpu.sh --resume
```

默认输出根目录：
`/share-mixed/project/robot-rfm/robot-exp/usr/ejw/posttrain/outputs/omnisf_composite_seen`。

每次启动都会建立独立的 `training_runs/<EXP_NAME>/<UTC时间>_<PID>/`，保存：

- `train.log`：训练进程完整 stdout、stderr。
- `metrics.csv`：打印的 step、action loss、梯度范数和吞吐等指标，逐条落盘。
- `launch.json`：实际命令、配置、数据/权重路径、统计文件校验值、PID。
- `status.json`：运行状态和最近指标，失败时记录退出码。

沿用原训练器每 10 步打印一次的行为：CSV 中的 loss 是训练器打印的区间平均值，
保留 4 位小数，并非每一步的原始 loss。正常最后一条为 step 119990；
最终 checkpoint 119999 不额外打印最后 9 步的平均 loss。关闭 W&B 不影响这些本地日志。
重新运行使用新日志目录，不覆盖已有日志；`LOG_DIR` 可指定尚不存在的新目录。

本子集的统计脚本读取全部 Parquet 数值列，使用 50 步动作窗口和末帧重复填充的
权重计算动作分布，以 float64 计算均值/标准差和精确加权分位数。
与 mixed 运行采用同一种全量统计方法；与原生统计脚本的区别是精确分位数替代
5000 个直方图区间近似。生成时验证 LeRobot 原生窗口索引和输入转换，
保存 `norm_stats.provenance.json` 记录数据覆盖范围、方法和校验结果。
