# Activate execution environment
source .venv/bin/activate

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.



# Optional: Generate plots after training
echo "Generating plots..."
# Read game from config (simple grep hack to avoid jq dependency)
GAME=$(grep '"game":' config.json | cut -d '"' -f 4)

PLOT_TAG="${PLOT_TAG:-default}"
python src/stable_baseline/main.py --mode plot --env $GAME --plot-tag $PLOT_TAG
