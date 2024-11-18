import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import wandb
import matplotlib

# Initialize the wandb API
api = wandb.Api()

# Specify your entity (username or team) and project name
entity = 'edenlum'  # Replace with your wandb username or team name
project = 'skip_layer_gpt2_wikitext'  # Replace with your wandb project name

# Get the artifact from run 1
artifact1 = api.artifact(f'{entity}/{project}/per_token_perplexities_0:v3', type='per_token_data')
artifact_dir1 = artifact1.download(root='run1_artifacts')

# Get the artifact from run 2
artifact2 = api.artifact(f'{entity}/{project}/per_token_perplexities_0:v2', type='per_token_data')
artifact_dir2 = artifact2.download(root='run2_artifacts')

# Now you can load the .pt files from the directories
data_run1 = torch.load(f'{artifact_dir1}/per_token_perplexities_0.pt')
data_run2 = torch.load(f'{artifact_dir2}/per_token_perplexities_0.pt')

perplexities_run1 = data_run1
perplexities_run2 = data_run2
# Compute the difference in per-token perplexities
perplexity_diff = perplexities_run2 - perplexities_run1

matplotlib.use('TkAgg')
# Plotting per-token perplexity differences (subset of tokens)
num_tokens_to_plot = 100  # Adjust as needed
plt.figure(figsize=(15, 5))
plt.bar(range(num_tokens_to_plot), perplexity_diff.numpy()[:num_tokens_to_plot])
plt.xlabel('Tokens')
plt.ylabel('Perplexity Difference (Run2 - Run1)')
plt.title('Per-Token Perplexity Differences for First 100 Tokens')
plt.tight_layout()
plt.savefig()

# Plot histogram of per-token perplexity differences
plt.figure(figsize=(10, 6))
plt.hist(perplexity_diff.numpy(), bins=100)
plt.xlabel('Per-Token Perplexity Difference (Run2 - Run1)')
plt.ylabel('Frequency')
plt.title('Histogram of Per-Token Perplexity Differences')
plt.show()

plt.plot([1,2,3], [4,5,6])
plt.show()