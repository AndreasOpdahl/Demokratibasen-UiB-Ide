# files needed (scp file_name olivia:):
# .env
# run_finetune.sbatch
# scripts/finetune_olivia.py
# data/output/processed_data_*.jsonl

mkdir -p logs
sbatch run_finetune.sbatch
squeue -u $USER
# tail logs as it runs/finishes
tail -f logs/gpu-smoke-*.out


salloc -p accel --gres=gpu:1 --cpus-per-task=4 --mem=16G -A nn12075k --time=01:00:00
