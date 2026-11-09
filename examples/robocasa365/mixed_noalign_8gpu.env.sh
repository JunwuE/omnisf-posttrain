# Shared storage defaults for the single-node mixed training launcher.
# Existing environment variables take precedence. No files are created here.
export ROBOCASA365_DATA_ROOT=${ROBOCASA365_DATA_ROOT:-/share-mixed/project/robot-rfm/robot-raw-data/robocasa365_lerobot2}
export ROBOCASA365_SOURCE_DATA_ROOT=${ROBOCASA365_SOURCE_DATA_ROOT:-/share-mixed/project/robot-rfm/robot-raw-data/robocasa365}
export ROBOCASA365_ASSETS_DIR=${ROBOCASA365_ASSETS_DIR:-/share/project/robot-rfm/robot-exp/usr/ejw/posttrain/outputs/omnisf_mixed/assets}
export ROBOCASA365_CHECKPOINT_DIR=${ROBOCASA365_CHECKPOINT_DIR:-/share/project/robot-rfm/robot-exp/usr/ejw/posttrain/outputs/omnisf_mixed/checkpoints}
export ROBOCASA365_PRETRAIN_PARAMS=${ROBOCASA365_PRETRAIN_PARAMS:-/share/project/robot-model/omnisf/35999/omnisf_9.0_jax_35999/params}
export ROBOCASA365_MIXED_WEIGHTS=${ROBOCASA365_MIXED_WEIGHTS:-1,1,1}
# Reuse the prepared tokenizer across nodes instead of downloading into /root/.cache.
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/share/project/robot-rfm/cache/openpi}
