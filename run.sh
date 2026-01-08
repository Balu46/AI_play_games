#!/bin/bash

# Activate execution environment
source .venv/bin/activate

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Run Training
USE_OPTIMIZED_PARAMS=1

help() {
  echo "Usage: ./run.sh [-n]"
  echo "  -n : Do NOT use optimized hyperparameters (default: use optimized if present)"
  exit 1
}

while getopts "nh" opt; do
  case "$opt" in
    n ) USE_OPTIMIZED_PARAMS=0 ;;
    h ) help ;;
    ? ) help ;;
  esac
done

echo "Starting training based on config.json..."
TRAIN_ARGS=(--mode train)
if [ "$USE_OPTIMIZED_PARAMS" = "0" ]; then
  TRAIN_ARGS+=(--no-optimized-params)
else
  echo "Using optimized hyperparameters from best_params.json (if present)."
fi
python src/stable_baseline/main.py "${TRAIN_ARGS[@]}"

# Optional: Generate plots after training
echo "Generating plots..."
# Read game from config (simple grep hack to avoid jq dependency)
GAME=$(grep '"game":' config.json | cut -d '"' -f 4)

python src/stable_baseline/main.py --mode plot --env $GAME
