#!/usr/bin/env bash
# Example script for running end-to-end tests on Olivia
# 
# This script demonstrates how to submit the test_e2e.sbatch job
# with various configurations.
#
# Usage:
#   chmod +x run_test_e2e_example.sh
#   ./run_test_e2e_example.sh
#
# Or copy the commands and run them directly on the server.

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

# Your SLURM account (REQUIRED - replace with your actual account)
SLURM_ACCOUNT="YOUR_PROJECT_ID"  # <-- CHANGE THIS to your account

# Model to test (default: gemma-2b - smallest and fastest)
MODEL="gemma-2b"

# Optional: Directory to keep test outputs (leave empty for temp directory)
TEST_DIR=""  # e.g., "/cluster/work/projects/YOUR_PROJECT_ID/ajayv/test_outputs"

# Test options (set to "true" to enable)
SKIP_TRAINING="false"      # Skip training test (only test utilities and evaluation)
SKIP_EVALUATION="false"    # Skip evaluation test
KEEP_TEST_DATA="false"     # Keep test datasets and outputs after testing

# ============================================================================
# EXAMPLE COMMANDS
# ============================================================================

echo "==================================================================="
echo "End-to-End Test Runner Examples"
echo "==================================================================="
echo ""

# Example 1: Basic test (default: gemma-2b, all tests)
echo "Example 1: Basic test with default settings"
echo "-------------------------------------------"
echo "sbatch --account=$SLURM_ACCOUNT \\"
echo "       run_test_e2e.sbatch \\"
echo "       --model=$MODEL"
echo ""

# Example 2: Test a specific model
echo "Example 2: Test a larger model (gemma-2-9b)"
echo "-------------------------------------------"
echo "sbatch --account=$SLURM_ACCOUNT \\"
echo "       run_test_e2e.sbatch \\"
echo "       --model=gemma-2-9b"
echo ""

# Example 3: Test with custom output directory
if [ -n "$TEST_DIR" ]; then
    echo "Example 3: Test with custom output directory"
    echo "-------------------------------------------"
    echo "sbatch --account=$SLURM_ACCOUNT \\"
    echo "       run_test_e2e.sbatch \\"
    echo "       --model=$MODEL \\"
    echo "       --test_dir=$TEST_DIR"
    echo ""
fi

# Example 4: Skip training (only test utilities and evaluation)
echo "Example 4: Skip training test (faster, if you only want to test evaluation)"
echo "-------------------------------------------"
echo "sbatch --account=$SLURM_ACCOUNT \\"
echo "       run_test_e2e.sbatch \\"
echo "       --model=$MODEL \\"
echo "       --skip_training"
echo ""

# Example 5: Keep test data for inspection
echo "Example 5: Keep test data after completion (for debugging)"
echo "-------------------------------------------"
echo "sbatch --account=$SLURM_ACCOUNT \\"
echo "       run_test_e2e.sbatch \\"
echo "       --model=$MODEL \\"
echo "       --keep_test_data"
echo ""

# Example 6: Dry-run (validate without submitting)
echo "Example 6: Dry-run (validate configuration without submitting job)"
echo "-------------------------------------------"
echo "sbatch --account=$SLURM_ACCOUNT \\"
echo "       run_test_e2e.sbatch \\"
echo "       --model=$MODEL \\"
echo "       --dry-run"
echo ""

# Example 7: Full test with all options
echo "Example 7: Full test with all options"
echo "-------------------------------------------"
echo "sbatch --account=$SLURM_ACCOUNT \\"
echo "       run_test_e2e.sbatch \\"
echo "       --model=$MODEL \\"
if [ -n "$TEST_DIR" ]; then
    echo "       --test_dir=$TEST_DIR \\"
fi
echo "       --keep_test_data"
echo ""

# ============================================================================
# QUICK START: Uncomment one of these to run immediately
# ============================================================================

# Uncomment the command you want to run:

# Basic test (recommended for first run):
# sbatch --account=$SLURM_ACCOUNT run_test_e2e.sbatch --model=gemma-2b

# Test with custom directory:
# sbatch --account=$SLURM_ACCOUNT run_test_e2e.sbatch --model=gemma-2b --test_dir=/path/to/test/outputs

# Test larger model:
# sbatch --account=$SLURM_ACCOUNT run_test_e2e.sbatch --model=gemma-2-9b

# ============================================================================
# MONITORING
# ============================================================================

echo "==================================================================="
echo "Monitoring Your Test Job"
echo "==================================================================="
echo ""
echo "After submitting, monitor with:"
echo "  squeue -u \$USER                    # Check job status"
echo "  tail -f logs/test-e2e-<JOB_ID>.out  # Watch output log"
echo "  tail -f logs/test-e2e-<JOB_ID>.err  # Watch error log"
echo ""
echo "Get your job ID from squeue output, then:"
echo "  JOB_ID=<your_job_id>"
echo "  tail -f logs/test-e2e-\$JOB_ID.out"
echo ""

# ============================================================================
# AVAILABLE MODELS
# ============================================================================

echo "==================================================================="
echo "Available Models for Testing"
echo "==================================================================="
echo ""
echo "Small models (fast, good for quick tests):"
echo "  - gemma-2b (recommended for first test)"
echo "  - gemma-7b"
echo "  - viking-7b"
echo "  - normistral-7b"
echo ""
echo "Medium models:"
echo "  - gemma-2-9b"
echo "  - viking-13b"
echo "  - normistral-11b"
echo "  - norskgpt-llama3-8b"
echo ""
echo "Large models (slower, use more memory):"
echo "  - gemma-2-27b"
echo "  - gemma-3-12b"
echo "  - gemma-3-27b"
echo "  - viking-33b"
echo "  - llama-2-13b-chat-norwegian"
echo ""

# ============================================================================
# ACTUAL SUBMISSION (uncomment to run)
# ============================================================================

# Uncomment the line below to actually submit the job:
# sbatch --account=$SLURM_ACCOUNT run_test_e2e.sbatch --model=$MODEL

echo "==================================================================="
echo "To run a test, uncomment one of the sbatch commands above"
echo "or copy a command from the examples and run it directly."
echo "==================================================================="
