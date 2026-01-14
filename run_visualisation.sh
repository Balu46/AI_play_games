#!/bin/bash

# Activate execution environment
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

# Default values
DEFAULT_ALGO="all"
# Try to get game from config, default to lunar_lander
DEFAULT_ENV=$(grep '"game":' config.json | cut -d '"' -f 4)
if [ -z "$DEFAULT_ENV" ]; then
    DEFAULT_ENV="lunar_lander"
fi

echo "--- Visualization Mode ---"
echo "Usage: ./run_visualisation.sh [algo] [env]"
echo "Available algorithms: ppo, a2c, dqn"
echo "Available environments: lunar_lander, car_racing, cart_pole"
echo ""

ALGO=${1:-$DEFAULT_ALGO}
ENV=${2:-$DEFAULT_ENV}

# echo "Plotting $ENV..."
# python src/stable_baseline/main.py --mode plot --env $ENV

echo "Visualizing $ALGO on $ENV..."
python src/stable_baseline/main.py --mode visualize --algo $ALGO --env $ENV
