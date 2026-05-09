source .venv/bin/activate \
&& VERSION=v17.1 && STEP=15000 \
&& VERSION2=v17.0 && STEP2=15000 \
&& uv run --no-sync scripts/extract_feature_vector.py \
  --a.config ${VERSION} \
  --a.dir checkpoints/${VERSION}/experiments/${STEP} \
  --b.config ${VERSION2} \
  --b.dir checkpoints/${VERSION2}/experiments/${STEP2} \
  --out checkpoints/diff/${VERSION}_${STEP}_minus_${VERSION2}_${STEP2}_ONLY_VLM.pth \
  --only-vlm \
  --strict-keys \
  --dtype fp32

