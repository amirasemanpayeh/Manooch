#!/bin/zsh
# Direct prompt generation bypass CLI - fastest method
# This script calls the StrategyManager directly without going through the interactive menu

echo "⚡ Direct prompt generation (bypassing CLI menu)..."
echo

# Activate virtual environment
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found. Please run ./run.sh first to set it up."
    exit 1
fi

source $VENV_DIR/bin/activate

# Hardcoded file paths
STORY_PATH="storage/shorts_publicaiton_folder/RedFlagRadio-n2w/shorts/rf_source_story.json"
STRATEGY_PATH="storage/shorts_publicaiton_folder/RedFlagRadio-n2w/strategy.json"

# Check if files exist
if [ ! -f "$STORY_PATH" ]; then
    echo "❌ Story file not found: $STORY_PATH"
    exit 1
fi

if [ ! -f "$STRATEGY_PATH" ]; then
    echo "❌ Strategy file not found: $STRATEGY_PATH"
    exit 1
fi

echo "📁 Using files:"
echo "   Story: $STORY_PATH"
echo "   Strategy: $STRATEGY_PATH"
echo

# Run Python script that calls the function directly
python3 -c "
import sys
import os
sys.path.append(os.getcwd())

from logic.shorts_strategy_manager import StrategyManager

try:
    print('🚀 Generating prompt...')
    manager = StrategyManager()
    prompt = manager.generate_prompt_from_story_file('$STORY_PATH', '$STRATEGY_PATH')
    
    print()
    print('=' * 80)
    print('GENERATED PROMPT')
    print('=' * 80)
    print(prompt)
    print('=' * 80)
    print()
    print('✅ Prompt generated successfully!')
    
except Exception as e:
    print(f'❌ Error generating prompt: {e}')
    sys.exit(1)
"