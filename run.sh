#!/bin/bash

# Activate execution environment
source .venv/bin/activate

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Run Training
echo "Starting training based on config.json..."
python src/stable_baseline/main.py --mode train

# Optional: Generate plots after training
echo "Generating plots..."
# Read game from config (simple grep hack to avoid jq dependency)
GAME=$(grep '"game":' config.json | cut -d '"' -f 4)

python src/stable_baseline/main.py --mode plot --env $GAME
