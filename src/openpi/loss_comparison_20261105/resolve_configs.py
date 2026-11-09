"""Reconstruct configs from recorded atomic CLI / current mixed launcher; no training."""

import dataclasses
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


HERE = Path(__file__).resolve().parent
OUTPUTS = Path('/share/project/robot-rfm/robot-exp/usr/ejw/posttrain/outputs')
PROJECT = Path('/home/junwue/code/omnisf')
ATOMIC_RUN = OUTPUTS / 'omnisf_atomic/noalign_runs/robocasa365_atomic_noalign_4gpu_20261025T081627Z'


def json_value(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_value(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (str, float, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def child(name):
    os.environ['ROBOCASA365_ASSETS_DIR'] = str(OUTPUTS / f'omnisf_{name}/assets')
    os.environ['ROBOCASA365_CHECKPOINT_DIR'] = str(OUTPUTS / f'omnisf_{name}/checkpoints')
    os.environ['ROBOCASA365_PRETRAIN_PARAMS'] = '/share/project/robot-model/omnisf/35999/omnisf_9.0_jax_35999/params'
    os.environ['ROBOCASA365_DATA_ROOT'] = (
        '/share/project/robot-rfm/robot-raw-data/robocasa365_lerobot2' if name == 'atomic' else
        '/share-mixed/project/robot-rfm/robot-raw-data/robocasa365_lerobot2')
    os.environ['ROBOCASA365_MIXED_WEIGHTS'] = '1,1,1'
    os.environ['PROJECT_ROOT'] = str(PROJECT)
    if name == 'atomic':
        command = json.loads((ATOMIC_RUN / 'launch.json').read_text())['command']
        provenance = 'Historical launch.json command, recorded environment paths, and current base config.'
    else:
        os.environ['EXP_NAME'] = 'robocasa365_target_mixed_noalign_120k_bs512_8gpu'
        command = subprocess.check_output(
            ['bash', str(PROJECT / 'scripts/train_robocasa365_mixed_noalign_8gpu.sh'), '--dry-run'], text=True).strip()
        provenance = ('Current mixed launcher defaults and base config, consistent with recovered log target/batch. '
                      'Original startup log is unavailable; additional historical CLI/env overrides cannot be excluded.')
    tokens = shlex.split(command)
    sys.argv = ['resolve_configs.py', *tokens[tokens.index('scripts/train.py') + 1:]]
    from openpi.training import config
    cfg = config.cli()
    assert not cfg.align_enabled and cfg.align_loss_coeff == 0 and cfg.vggt_weight_path is None
    exported = {'reconstructed_at_utc': datetime.now(timezone.utc).isoformat(),
                'provenance': provenance, 'command': command, 'config': json_value(cfg),
                'effective_batch_size': cfg.effective_batch_size}
    (HERE / f'{name}_config.json').write_text(json.dumps(exported, indent=2) + '\n')
    print(json.dumps({'config': cfg.name, 'steps': cfg.num_train_steps, 'batch': cfg.batch_size,
                      'fsdp': cfg.fsdp_devices, 'align': cfg.align_enabled}))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        child(sys.argv[1])
    else:
        # Config modules read path variables at import time; use one process per recipe.
        for recipe in ('atomic', 'mixed'):
            subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), recipe], check=True)
