mkdir -p logs
sbatch run_test.sbatch
squeue -u $USER
# tail logs as it runs/finishes
tail -f logs/gpu-smoke-*.out
