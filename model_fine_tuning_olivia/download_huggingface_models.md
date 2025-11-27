Save GPU time by downloading models on login node.

On ARM GPU nodes, make sure:
"The model download happens with the same PyTorch version you'll use on GPU nodes
The cache directory (~/.cache/huggingface/) is accessible from both login and GPU nodes (usually is by default)"

TODO: sharing base models with other project users

# Install if needed
pip install huggingface-hub

# Login once
huggingface-cli login

# Download model
huggingface-cli download google/gemma-2b

# Or for gated models with token
huggingface-cli download google/gemma-2b --token YOUR_TOKEN