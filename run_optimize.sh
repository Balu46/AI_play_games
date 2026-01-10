#!/bin/bash

# Activate execution environment
source .venv/bin/activate

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Default values
ENV="all"
ALGO="all"
TRIALS=0
TIMESTEPS=0

# Supported lists
ALL_ENVS=("cart_pole" "lunar_lander" "car_racing")
ALL_ALGOS=("dqn" "ppo" "a2c")

# Help function
help() {
   echo "Usage: ./run_optimize.sh [ -e environment ] [ -a algorithm ] [ -n trials ] [ -t timesteps ]"
   echo "  -e : Environment (default: all - runs on ${ALL_ENVS[*]})"
   echo "  -a : Algorithm (default: all - runs ${ALL_ALGOS[*]})"
   echo "  -n : Number of trials (default: env-specific)"
   echo "  -t : Timesteps per trial (default: env-specific)"
   exit 1
}

# Parse arguments
while getopts "e:a:n:t:h" opt; do
   case "$opt" in
      e ) ENV="$OPTARG" ;;
      a ) ALGO="$OPTARG" ;;
      n ) TRIALS="$OPTARG" ;;
      t ) TIMESTEPS="$OPTARG" ;;
      h ) help ;;
      ? ) help ;;
   esac
done

# Determine environments to run
if [ "$ENV" == "all" ]; then
    TARGET_ENVS=("${ALL_ENVS[@]}")
else
    # Allow comma separated or just single
    TARGET_ENVS=($ENV) 
fi

# Determine algorithms to run
if [ "$ALGO" == "all" ]; then
    TARGET_ALGOS=("${ALL_ALGOS[@]}")
else
    TARGET_ALGOS=($ALGO)
fi

echo "Detailed Configuration:"
echo "  Environments: ${TARGET_ENVS[*]}"
echo "  Algorithms:   ${TARGET_ALGOS[*]}"
if [ "$TRIALS" -gt 0 ] && [ "$TIMESTEPS" -gt 0 ]; then
   echo "  Trials:       $TRIALS"
   echo "  Timesteps:    $TIMESTEPS"
else
   echo "  Trials:       env-specific defaults"
   echo "  Timesteps:    env-specific defaults"
fi
echo "--------------------------------------------------------"

# Loop and run
for env in "${TARGET_ENVS[@]}"; do
    # Environment-specific defaults (can be overridden by -n/-t)
    env_trials="$TRIALS"
    env_timesteps="$TIMESTEPS"
    if [ "$env_trials" -eq 0 ] || [ "$env_timesteps" -eq 0 ]; then
        case "$env" in
            cart_pole)
                env_trials=40
                env_timesteps=150000
                ;;
            lunar_lander)
                env_trials=60
                env_timesteps=300000
                ;;
            car_racing)
                env_trials=100
                env_timesteps=800000
                ;;
            *)
                env_trials=40
                env_timesteps=150000
                ;;
        esac
    fi
    for algo in "${TARGET_ALGOS[@]}"; do
        echo "========================================================"
        echo "Running optimization for $algo on $env..."
        echo "========================================================"
        
        # Run the optimization
        # We use a subprocess so that failure in one doesn't kill the whole script (unless we want it to?)
        # Adding a small sleep or check might be nice, but simple logic first.
        
        python3 src/stable_baseline/main.py --mode optimize --env "$env" --algo "$algo" --n-trials "$env_trials" --timesteps "$env_timesteps"
        
        RET_CODE=$?
        if [ $RET_CODE -ne 0 ]; then
            echo "!!! Optimization failed for $algo on $env with exit code $RET_CODE !!!"
            # Optional: exit 1 # Uncomment to stop on first failure
        else
            echo "Successfully optimized $algo on $env"
        fi
        
        echo ""
    done
done

echo "Batch optimization complete."
