#!/bin/bash

# Script description
# Author: Kshitz Kaushik    
# Date: 2025-04-09

# Exit on error
set -e

# Define variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Functions
function cleanup() {
    # Add cleanup tasks here
    echo "Performing cleanup..."
}

# Trap errors
trap cleanup EXIT

# Main script logic
function main() {
    echo "Installing dependencies..."
    pip install -r requirements.txt
    huggingface-cli login
    wandb login
    echo "Training LLama3.2-instruct-3B ..."
    echo "Fetching dataset..."
    mkdir -p training_data
    wget --retry-connrefused --tries=3 --progress=bar -P training_data https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V4.3_unfiltered_cleaned_split.json
    cd "$SCRIPT_DIR"
    echo "Generating Training Data ..."
    python "eagle/ge_data/ge_data_all_llama3instruct.py"
    echo "Saving Model ..."
    python "eagle/save_model.py" 
    echo "Training ..."
    accelerate launch -m --mixed_precision=bf16 eagle.train.main --tmpdir outdir0/1/ --cpdir eagle/llama3.2-3b-instruct-local/ --configpath eagle/train/llama_3-2_instruct_3B_config.json

}

# Run main function
main "$@"