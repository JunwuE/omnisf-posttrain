# Independent outputs for composite_unseen; exported paths take precedence.
export ROBOCASA365_OUTPUT_ROOT=${ROBOCASA365_OUTPUT_ROOT:-/share-mixed/project/robot-rfm/robot-exp/usr/ejw/posttrain/outputs/omnisf_composite_unseen}
export ROBOCASA365_DATA_ROOT=${ROBOCASA365_DATA_ROOT:-/share-mixed/project/robot-rfm/robot-raw-data/robocasa365_lerobot2}
export ROBOCASA365_ASSETS_DIR=${ROBOCASA365_ASSETS_DIR:-$ROBOCASA365_OUTPUT_ROOT/assets}
export ROBOCASA365_CHECKPOINT_DIR=${ROBOCASA365_CHECKPOINT_DIR:-$ROBOCASA365_OUTPUT_ROOT/checkpoints}
export ROBOCASA365_PRETRAIN_PARAMS=${ROBOCASA365_PRETRAIN_PARAMS:-/share/project/robot-model/omnisf/35999/omnisf_9.0_jax_35999/params}
export ROBOCASA365_VGGT_PATH=${ROBOCASA365_VGGT_PATH:-/share/project/robot-model/omnisf/components/omnivggt}
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/share/project/robot-rfm/cache/openpi}
