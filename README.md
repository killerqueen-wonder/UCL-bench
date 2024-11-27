# UCL-bench



User-Centric Legal benchmark

# Dataset

## prompt
/dataset/data_processing/legal_prompt.json shows the user-simulator prompt, model prompt, and evaluation prompt for each task. Due to the different backgrounds of each task, their prompts have slight variations.

## data
We divide the data into two tracks: **public** and **private**. The data for the **public track** is openly available (see `/dataset/legal_data_sample.json`), while we reserve a portion of the data for the **private track**. In the private track, we evaluate existing models in an offline manner (models are downloaded and evaluated locally). This approach helps us avoid data leakage.

# Experiment

## multi-turn dialogue construction
local_inference.py and api_inference.py demonstrate our approach to building multi-turn dialogues. The api_inference.py script achieves this by calling the model API, while local_inference.py accomplishes it by downloading the model weights to the local server for inference.

## evaluation
evaluate.py demonstrates the method of invoking GPT-4 for evaluation.
