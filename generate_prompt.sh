#!/bin/zsh
# Generate prompt from story using hardcoded paths
# This is a convenience script for quick testing

echo "🚀 Generating prompt from story with hardcoded paths..."
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

# Use printf to feed the inputs to the interactive CLI
printf "2\n$STORY_PATH\n$STRATEGY_PATH\nq\n" | python main.py