#!/bin/zsh
# Direct video generation bypass CLI - fastest method
# This script calls the StrategyManager directly without going through the interactive menu

echo "⚡ Direct video generation (bypassing CLI menu)..."
echo

# Activate virtual environment
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found. Please run ./run.sh first to set it up."
    exit 1
fi

source $VENV_DIR/bin/activate

# Hardcoded file path
PLAN_PATH="storage/shorts_publicaiton_folder/RedFlagRadio-n2w/shorts/rf_0002.json"

# Check if file exists
if [ ! -f "$PLAN_PATH" ]; then
    echo "❌ Plan file not found: $PLAN_PATH"
    exit 1
fi

echo "📁 Using file:"
echo "   Plan: $PLAN_PATH"
echo

# Run Python script that calls the function directly
python3 -c "
import sys
import os
sys.path.append(os.getcwd())

from logic.shorts_strategy_manager import StrategyManager

try:
    print('🚀 Generating video from plan...')
    manager = StrategyManager()
    plan = manager.generate_video_from_plan_file('$PLAN_PATH')
    
    print()
    print('=' * 80)
    print('VIDEO GENERATION COMPLETED')
    print('=' * 80)
    print(f'Total shots processed: {len(plan.shots)}')
    print(f'Shots ordered: {[shot.order for shot in plan.ordered()]}')
    if plan.video_url:
        print(f'Video URL: {plan.video_url}')
    print('=' * 80)
    print()
    print('✅ Video generated successfully!')
    
except Exception as e:
    print(f'❌ Error generating video: {e}')
    sys.exit(1)
"