"""Create audited loss plots and a config comparison from original training logs."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUTPUTS = Path('/share/project/robot-rfm/robot-exp/usr/ejw/posttrain/outputs')
PROJECT = Path('/home/junwue/code/omnisf')
ATOMIC_RUN = OUTPUTS / 'omnisf_atomic/noalign_runs/robocasa365_atomic_noalign_4gpu_20261025T081627Z'
MIXED_RECOVERY = OUTPUTS / 'omnisf_mixed/recovered_logs/20261105T052102Z'
LOGS = {'atomic': ATOMIC_RUN / 'train.log', 'mixed': MIXED_RECOVERY / 'tmux_pane_0.log'}
PIDS = {'atomic': 116955, 'mixed': 3104}
PATTERN = re.compile(r'^\s*(\d\d:\d\d:\d\d\.\d+) \[I\] Step (\d+): (.*?)\s+\((\d+):train\.py:(\d+)\)\s*$')
VALUE_PATTERN = re.compile(r'\b([a-z_]+)=([-+0-9.eE]+)(?=,|\s|$)')


def read_metrics(name):
    rows = []
    for line_number, line in enumerate(LOGS[name].read_text().splitlines(), 1):
        match = PATTERN.match(line)
        if not match or int(match[4]) != PIDS[name]:
            continue
        values = {key: float(value) for key, value in VALUE_PATTERN.findall(match[3])}
        if 'loss' not in values:
            continue
        assert 'align_loss' not in values and 'action_loss' not in values
        step = int(match[2])
        row = {'step': step, 'action_loss': values['loss'], 'loss': values['loss'],
               'window_first_step': max(0, step - 9), 'window_last_step': step,
               'log_time_without_date': match[1], 'source_line': line_number,
               'source_file': str(LOGS[name]), 'pid': int(match[4])}
        row.update({('learning_rate_rounded_log' if key == 'learning_rate' else key): value
                    for key, value in values.items() if key != 'loss'})
        rows.append(row)
    steps = [r['step'] for r in rows]
    expected = list(range(0, 60000, 10)) if name == 'atomic' else list(range(111130, 120000, 10))
    assert steps == expected, (name, len(steps), steps[:3], steps[-3:])
    assert all(np.isfinite(row['loss']) and row['loss'] > 0 for row in rows)
    with (HERE / f'{name}_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def describe(rows):
    result = {'records': len(rows), 'first_logged_step': rows[0]['step'],
              'first_logged_loss': rows[0]['loss'], 'last_logged_step': rows[-1]['step'],
              'last_logged_loss': rows[-1]['loss'], 'last_source_line': rows[-1]['source_line']}
    for n in (10, 100, 500):
        tail = rows[-n:]
        values = [r['loss'] for r in tail]
        result[f'last_{n * 10}_logged_steps'] = {
            'records': len(tail), 'first_window_step': tail[0]['window_first_step'],
            'last_window_step': tail[-1]['step'], 'mean': statistics.mean(values),
            'min': min(values), 'max': max(values), 'std_population': statistics.pstdev(values)}
    result['milestones'] = {str(row['step']): row['loss'] for row in rows
                            if row['step'] in (0, 10, 100, 500, 1000, 5000, 10000, 20000,
                                                30000, 40000, 50000, 59990, 119990)}
    return result


def draw_curves(records, stats):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 14, 'axes.titleweight': 'bold',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#94a3b8', 'text.color': '#172554',
                         'axes.labelcolor': '#334155', 'xtick.color': '#475569',
                         'ytick.color': '#475569', 'svg.fonttype': 'none',
                         'pdf.fonttype': 42})
    colors = {'atomic': '#2563eb', 'mixed': '#ea580c'}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.8))
    fig.subplots_adjust(left=.075, right=.965, bottom=.115, top=.855, hspace=.40, wspace=.23)
    fig.suptitle('Atomic vs. mixed: action training loss', x=.075, y=.968, ha='left', fontsize=23, fontweight='bold')
    fig.text(.075, .926, 'Both evaluated checkpoints use no alignment. Atomic: complete logged history; mixed: recovered tail only.', fontsize=11.5)

    def style(ax):
        ax.grid(axis='y', color='#e2e8f0', linewidth=.8)
        ax.set_axisbelow(True)
        ax.set_xlabel('Optimizer step (zero-based)')
        ax.set_ylabel('Action loss')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000:g}k' if x else '0'))

    def curve(ax, name, rows, window=50):
        x = np.array([r['step'] for r in rows])
        y = np.array([r['loss'] for r in rows])
        ax.plot(x, y, color=colors[name], alpha=.25, linewidth=.65, label='Logged 10-step mean')
        rolling = pd.Series(y).rolling(window, min_periods=window).mean().to_numpy()
        ax.plot(x, rolling, color=colors[name], linewidth=2.2, label='500-step moving average')
        return x, y

    for ax in axes.flat:
        style(ax)
    curve(axes[0, 0], 'atomic', records['atomic'])
    axes[0, 0].set_title('A  Atomic: full logged history', loc='left', pad=13)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlim(0, 60000)
    axes[0, 0].set_ylim(.0035, 3.2)
    axes[0, 0].set_ylabel('Action loss (log scale)')
    axes[0, 0].legend(frameon=False, loc='upper right', fontsize=9)
    axes[0, 0].text(.98, .49, '2.4170 at step 0\n0.0047 at step 59,990',
                    transform=axes[0, 0].transAxes, ha='right', color=colors['atomic'])

    ax = axes[0, 1]
    ax.set_title('B  Mixed: recovered coverage', loc='left', pad=13)
    ax.axvspan(0, 111130, facecolor='#f1f5f9', hatch='///', edgecolor='#e2e8f0', linewidth=0)
    curve(ax, 'mixed', records['mixed'])
    ax.set_xlim(0, 120000)
    ax.set_ylim(.004, .008)
    ax.set_yticks([.004, .005, .006, .007, .008])
    ax.text(.43, .52, 'No recovered logs\n0–111,120\nNo curve inferred',
            transform=ax.transAxes, ha='center', va='center', fontsize=13, color='#64748b',
            bbox={'facecolor': '#f1f5f9', 'edgecolor': 'none', 'pad': 8})
    ax.annotate('887 records\n111,130–119,990', xy=(116000, .0066), xytext=(58000, .00755),
                color=colors['mixed'], fontsize=10,
                arrowprops={'arrowstyle': '->', 'color': colors['mixed']})

    for name, ax in zip(('atomic', 'mixed'), axes[1]):
        rows = records[name][-887:]
        x, y = curve(ax, name, rows)
        ax.set_title(('C  Atomic' if name == 'atomic' else 'D  Mixed') + ': final logged segment', loc='left', pad=13)
        ax.set_xlim(x[0] - 150, x[-1] + 500)
        ax.set_ylim(.004, .0076)
        ax.set_yticks([.004, .005, .006, .007])
        mean = stats[name]['last_1000_logged_steps']['mean']
        ax.hlines(mean, x[-100], x[-1], colors='#0f172a', linestyles='--', linewidth=1.5)
        ax.text(.03, .92, f'Last 1,000 logged steps: mean {mean:.6f}',
                transform=ax.transAxes, color=colors[name], fontsize=10.5,
                bbox={'facecolor': 'white', 'alpha': .92, 'edgecolor': 'none', 'pad': 4})
        ax.scatter([x[-1]], [y[-1]], s=30, color=colors[name], zorder=5)
        ax.text(.97, .08, f'Last logged loss: {y[-1]:.4f}', transform=ax.transAxes,
                ha='right', color=colors[name], fontsize=10)
    fig.text(.075, .044, 'Light lines: rounded training logs. Dark lines: 50-record moving average. Bottom panels share the same y-axis scale.', fontsize=9.5, color='#475569')
    fig.text(.075, .023, 'Datasets and quantile normalization differ. These are training losses, not validation losses or success rates. Final 9 updates have no loss log.', fontsize=9.5, color='#475569')
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(HERE / f'loss_curves.{extension}', dpi=200, facecolor='white')
    plt.close(fig)


def make_config_table(configs, mixture):
    a, m = (configs[name]['config'] for name in ('atomic', 'mixed'))
    rows = [
        ('配置名', a['name'], m['name']),
        ('训练入口', 'scripts/train.py', 'scripts/train.py'),
        ('模型', 'π0.5 / gemma_2b + gemma_300m', 'π0.5 / gemma_2b + gemma_300m'),
        ('微调方式', '全参数微调，未使用 LoRA', '全参数微调，未使用 LoRA'),
        ('初始化权重', 'OmniSF 35999', 'OmniSF 35999'),
        ('数据', '仅 atomic', 'atomic + composite_seen + composite_unseen'),
        ('采样权重', '1', '1,1,1；实际概率按 weight × frames'),
        ('预期采样占比', 'atomic 100%', ' / '.join(f"{s['sampling_probability']:.2%}" for s in mixture)),
        ('训练步数', a['num_train_steps'], m['num_train_steps']),
        ('GPU / FSDP', a['fsdp_devices'], m['fsdp_devices']),
        ('全局 batch', a['batch_size'], m['batch_size']),
        ('每卡 batch', a['batch_size'] // a['fsdp_devices'], m['batch_size'] // m['fsdp_devices']),
        ('梯度累积', a['gradient_accumulation_steps'], m['gradient_accumulation_steps']),
        ('总样本抽取量（不去重）', a['num_train_steps'] * a['batch_size'], m['num_train_steps'] * m['batch_size']),
        ('Alignment', '关闭；系数 0；不加载 VGGT', '关闭；系数 0；不加载 VGGT'),
        ('优化器', 'AdamW β1=0.9, β2=0.95, ε=1e-8', 'AdamW β1=0.9, β2=0.95, ε=1e-8'),
        ('Weight decay / 梯度裁剪', '1e-10 / 1.0', '1e-10 / 1.0'),
        ('学习率', 'cosine: 2.5e-5 → 2.5e-6', 'cosine: 2.5e-5 → 2.5e-6'),
        ('Warmup', a['lr_schedule']['warmup_steps'], m['lr_schedule']['warmup_steps']),
        ('学习率 decay_steps', a['lr_schedule']['decay_steps'], m['lr_schedule']['decay_steps']),
        ('EMA', a['ema_decay'], m['ema_decay']),
        ('State dropout', a['state_dropout_prob'], m['state_dropout_prob']),
        ('随机种子', a['seed'], m['seed']),
        ('动作 horizon / 模型维度', f"{a['model']['action_horizon']} / {a['model']['action_dim']}", f"{m['model']['action_horizon']} / {m['model']['action_dim']}"),
        ('归一化', 'atomic quantile norm', '三个子集共用 mixed quantile norm'),
        ('数据读取线程/worker 设置', a['num_workers'], m['num_workers']),
        ('记录间隔', a['log_interval'], m['log_interval']),
        ('Checkpoint 间隔', a['save_interval'], m['save_interval']),
        ('Checkpoint 保留', '最新；最终 59999', '最新 + 每 20000 步；最终 119999'),
        ('W&B', '关闭', '关闭'),
    ]
    with (HERE / 'config_comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['参数', 'Atomic noalign', 'Mixed noalign'])
        writer.writerows(rows)
    return rows


def main():
    records = {name: read_metrics(name) for name in LOGS}
    stats = {name: describe(rows) for name, rows in records.items()}
    configs = {name: json.loads((HERE / f'{name}_config.json').read_text()) for name in LOGS}
    historical_path = OUTPUTS / 'robocasa365_eval/success_diagnosis_20261105T041532Z/historical_comparison.json'
    historical = json.loads(historical_path.read_text())
    for name, key in [('atomic', 'old'), ('mixed', 'mixed')]:
        checkpoint = historical[key]['server']['checkpoint']
        assert 'noalign' in checkpoint
        norm = Path(historical[key]['server']['norm_stats'])
        assert hashlib.sha256(norm.read_bytes()).hexdigest() == historical[key]['server']['norm_sha256']
        stats[name]['checkpoint'] = checkpoint
        stats[name]['norm_sha256'] = historical[key]['server']['norm_sha256']
    norm_audit_path = OUTPUTS / 'omnisf_mixed/norm_stats_runs/20261029T075022Z/validation.json'
    norm_audit = json.loads(norm_audit_path.read_text())
    components = norm_audit['components']
    total = sum(c['frames'] * c['weight'] for c in components)
    mixture = [{**{key: c[key] for key in ('name', 'frames', 'episodes', 'weight')},
                'sampling_probability': c['frames'] * c['weight'] / total} for c in components]
    config_rows = make_config_table(configs, mixture)
    summary = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'runs': stats, 'mixture_from_recipe_and_preparation_metadata': mixture,
               'loss_definition': 'Action flow-matching squared error, averaged over model action dimensions, action horizon and batch; no alignment.',
               'logging_semantics': 'Regular records average 10 optimizer updates and are rounded to 4 decimal places. Step 0 averages one update. Final 9 updates are not logged.',
               'limitations': ['Mixed logs before step 111130 are unavailable.',
                               'Different training data and normalization; not a controlled model-quality comparison.',
                               'Config JSON files are reconstructed, not immutable original complete config dumps.'],
               'plot_moving_average_records': 50}
    (HERE / 'loss_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    draw_curves(records, stats)
    a, m = stats['atomic'], stats['mixed']
    lines = [
        'Atomic 与 mixed 训练 loss / config 对照（2026-11-05）', '',
        '本报告匹配已经用于评测的两个 checkpoint：atomic noalign 59999 和 mixed noalign 119999。',
        '更正此前将 atomic 的 alignment 实验日志用于解释评测模型的说法：atomic 目录里还有另一条 align 实验；它不属于这里的 atomic 59999，不能拼接进本曲线。两个目标实验都运行 train.py，只计算动作 flow-matching loss。', '',
        '![训练 loss 曲线](loss_curves.png)', '',
        '| 指标 | Atomic noalign | Mixed noalign |', '|---|---:|---:|',
        f"| 可用日志步数 | {a['first_logged_step']}–{a['last_logged_step']} | {m['first_logged_step']}–{m['last_logged_step']} |",
        f"| 记录数 | {a['records']} | {m['records']} |",
        f"| 首条可用 loss | {a['first_logged_loss']:.4f}（step 0） | {m['first_logged_loss']:.4f}（step 111130；非训练起点） |",
        f"| 最后一条 loss | {a['last_logged_loss']:.4f} @ {a['last_logged_step']} | {m['last_logged_loss']:.4f} @ {m['last_logged_step']} |",
        f"| 末尾 100 个已记录步均值 | {a['last_100_logged_steps']['mean']:.6f} | {m['last_100_logged_steps']['mean']:.6f} |",
        f"| 末尾 1000 个已记录步均值 | {a['last_1000_logged_steps']['mean']:.6f} | {m['last_1000_logged_steps']['mean']:.6f} |",
        f"| 末尾 5000 个已记录步均值 | {a['last_5000_logged_steps']['mean']:.6f} | {m['last_5000_logged_steps']['mean']:.6f} |", '',
        'atomic 从初期的 2.4170 逐步降至约 0.0047；mixed 只能看到末段，末段围绕约 0.0065 波动。缺少 mixed 前段，因此不能判断它的完整收敛速度或何时发生变化。',
        '常规点是最近 10 步的均值，不是单个 batch 的瞬时值；表中区间均值由已四舍五入的日志计算。step 0 仅包含首次更新。末尾 59991–59999 / 119991–119999 没有单独 loss 日志，因此不能称最后记录值为最终 checkpoint 的精确单步 loss。',
        '图 A 使用对数纵轴展示 atomic 完整下降过程；图 B 灰色区域是 mixed 未恢复区间，不插值、不补点；图 C/D 使用相同纵轴尺度展示各自末尾 887 条记录。浅线为原日志，深线为 50 条记录（约 500 步）移动平均。',
        '两个实验的数据分布和 quantile norm 不同，不能将 loss 差值直接解释成模型效果比例。这些是训练 loss，不是验证集 loss 或评测成功率。mixed 没有保存按三个子集分别统计的 loss，无法由总均值还原三条曲线。', '',
        '配置对照：', '', '| 参数 | Atomic noalign | Mixed noalign |', '|---|---|---|',
        *[f'| {key} | {av} | {mv} |' for key, av, mv in config_rows], '',
        '两组配置均由基础 Python 配置 + 启动脚本 CLI 覆盖 + 环境变量组合得到。基础文件里的默认 20000 步、batch 1、align=True 不代表最终运行参数。',
        f"atomic 使用原始 [launch.json]({ATOMIC_RUN / 'launch.json'}) 中的命令及记录的环境路径重建；mixed 使用当前启动脚本及准备记录重建，训练目标步数和 batch 与恢复日志一致。mixed 完整启动日志缺失，无法排除当时另有 CLI 或环境覆盖。", '',
        f"- Atomic 启动参数：[train_robocasa365_noalign_4gpu.sh]({PROJECT / 'scripts/train_robocasa365_noalign_4gpu.sh'})",
        f"- Mixed 启动参数：[train_robocasa365_mixed_noalign_8gpu.sh]({PROJECT / 'scripts/train_robocasa365_mixed_noalign_8gpu.sh'})",
        f"- 基础配置：[robocasa365_config.py]({PROJECT / 'src/openpi/training/misc/robocasa365_config.py'})",
        f"- Atomic 运行环境：[environment.sh]({ATOMIC_RUN / 'environment.sh'})",
        f"- Mixed 路径配置：[mixed_noalign_8gpu.env.sh]({PROJECT / 'examples/robocasa365/mixed_noalign_8gpu.env.sh'})", '',
        '混合采样：1,1,1 是源权重，不是三个子集各占三分之一。采样概率与 weight × frames 成正比，默认约为 atomic 14.92%、composite_seen 40.13%、composite_unseen 44.95%。这是配方期望，不是重建的实际逐样本抽样记录。',
        '日志里的 learning_rate=0.0000 是保留四位小数造成的显示结果；配置学习率量级为 1e-5 / 1e-6，不表示停止学习。', '',
        '导出文件：', '',
        '- loss_curves.png / .svg / .pdf：曲线图，SVG/PDF 可用于排版。',
        '- atomic_metrics.csv / mixed_metrics.csv：日志指标，包含原文件路径、行号和记录窗口。',
        '- atomic_config.json / mixed_config.json：解析后的配置及重建依据。',
        '- config_comparison.csv：可用电子表格打开的配置对照。',
        '- loss_summary.json：统计数据与限制说明。',
        '- source_manifest.json：证据文件和生成代码的 SHA256。', '',
        f"Atomic 原始日志：[train.log]({LOGS['atomic']})；Mixed 恢复日志：[tmux_pane_0.log]({LOGS['mixed']})。", '',
        '复现：先用具备 OmniSF 依赖的 Python 运行 resolve_configs.py（仅 CPU 配置解析），再用具备 matplotlib/numpy/pandas 的 Python 运行 build_report.py；均不会加载权重或启动训练。',
    ]
    (HERE / 'README.md').write_text('\n'.join(lines) + '\n')
    sources = [*LOGS.values(), ATOMIC_RUN / 'launch.json', ATOMIC_RUN / 'environment.sh',
               MIXED_RECOVERY / 'capture_manifest.json', historical_path, norm_audit_path,
               PROJECT / 'scripts/train_robocasa365_noalign_4gpu.sh',
               PROJECT / 'scripts/train_robocasa365_mixed_noalign_8gpu.sh',
               PROJECT / 'examples/robocasa365/mixed_noalign_8gpu.env.sh',
               PROJECT / 'src/openpi/training/misc/robocasa365_config.py',
               PROJECT / 'src/openpi/training/optimizer.py', PROJECT / 'scripts/train.py',
               PROJECT / 'src/openpi/models/pi0.py', PROJECT / 'src/openpi/training/data_loader.py',
               HERE / 'resolve_configs.py', HERE / 'build_report.py']
    manifest = [{'path': str(p), 'bytes': p.stat().st_size,
                 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]
    (HERE / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(HERE), 'atomic': stats['atomic'], 'mixed': stats['mixed']}, indent=2))


if __name__ == '__main__':
    main()
