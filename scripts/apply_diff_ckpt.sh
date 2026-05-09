VERSION=v30
source .venv/bin/activate \
&& BASE=/data/user/wsong890/lifuhao/openpi-main/checkpoints/pytorch_pi0_base/model.safetensors \
&& DIFF=/data/user/wsong890/lifuhao/openpi-main/checkpoints/diff/v16.1_30000_minus_v16.0_30000.pth \
&& uv run --no-sync scripts/apply_diff_ckpt.py \
  --base-safetensors ${BASE} \
  --diff-pth ${DIFF} \
  --out-safetensors /data/user/wsong890/lifuhao/openpi-main/checkpoints/vector_init_ckpt/${VERSION}/model.safetensors \
  --scale 1.0 \
  --no-strict-keys \
  --dtype fp32 \
  --device cpu

# Only in base (up to 30): [] Only in diff (up to 30): ['paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'] 也就是说语言模型里有个额外的embeds token

rsync -av --ignore-existing \
  /cpfs01/projects-HDD/cfff-377aad6b032c_HDD/chenshuai/wenxuan/lifuhao/openpi-main/checkpoints/pytorch_pi0_base/ \
  /cpfs01/projects-HDD/cfff-377aad6b032c_HDD/chenshuai/wenxuan/lifuhao/openpi-main/checkpoints/vector_init_ckpt/${VERSION}/
