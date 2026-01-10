#!/bin/bash

# Activate execution environment
source .venv/bin/activate

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Allow flag override
USE_OPTIMIZED_PARAMS="${USE_OPTIMIZED_PARAMS:-1}"
for arg in "$@"; do
  case "$arg" in
    -n|--no-optimized)
      USE_OPTIMIZED_PARAMS=0
      ;;
  esac
done

# Run Training
echo "Starting training based on config.json..."
TRAIN_ARGS=(--mode train)
if [ "$USE_OPTIMIZED_PARAMS" != "1" ]; then
  TRAIN_ARGS+=(--no-optimized-params)
else
  echo "Using optimized hyperparameters from best_params.json (if present)."
fi
python src/stable_baseline/main.py "${TRAIN_ARGS[@]}"

# Optional: Generate plots after training
echo "Generating plots..."
# Read game from config (simple grep hack to avoid jq dependency)
GAME=$(grep '"game":' config.json | cut -d '"' -f 4)
PLOT_TAG="default"
if [ "$USE_OPTIMIZED_PARAMS" = "1" ]; then
  PLOT_TAG="optimized"
fi

python src/stable_baseline/main.py --mode plot --env $GAME --plot-tag $PLOT_TAG
