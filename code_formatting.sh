#!/bin/bash

# List of directories and files to format
TARGETS=(
    "dataset_factory"
    "strategies"
    "losses"
    "models"
    "optimizers"
    "topologies"
    "utils"
    "main.py"
)

# Iterate over each target and apply isort and black
for TARGET in "${TARGETS[@]}"; do
    echo "Formatting $TARGET..."
    isort --profile=black "$TARGET"
    black "$TARGET"
done

echo "Formatting completed."