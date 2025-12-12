# Running sbatch on olivia

```sbatch
sbatch -J gemma-3-12b-pt-finetune \
	--ntasks=1 \
	--gres=gpu:4 \
	--cpus-per-task=8 \
	--mem=150G \
	--export=MODEL=gemma-3-12b-pt,TASK_LIMIT='--max_steps 90' \
    run_finetune_apptainer.sbatch
```


sbatch -J gemma-3-12b-pt-finetune --ntasks=1 --gres=gpu:2 --cpus-per-task=8 --mem=150G --export=MODEL=gemma-3-12b-pt,TASK_LIMIT='--max_steps 1000' --time=300 run_finetune_apptainer.sbatch



# Other models

google/gemma-7b
google/gemma-2-9b
google/gemma-2-27b
google/gemma-3-12b-pt (multimodal)
google/gemma-3-27b-pt (multimodal)
