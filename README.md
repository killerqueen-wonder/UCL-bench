# UCL-bench



User-Centric Legal benchmark

# Dataset

## prompt
/dataset/data_processing/legal_prompt.json shows the user-simulator prompt, model prompt, and evaluation prompt for each task. Due to the different backgrounds of each task, their prompts have slight variations.

## data
/dataset/legal_data_sample.json shows our data. We only open-source part of the data and keep some data private to prevent data leakage. We will download the model weights ourselves and conduct tests on the local server.

# Experiment

## multi-turn dialogue construction
local_inference.py and api_inference.py demonstrate our approach to building multi-turn dialogues. The api_inference.py script achieves this by calling the model API, while local_inference.py accomplishes it by downloading the model weights to the local server for inference.

## evaluation
evaluate.py demonstrates the method of invoking GPT-4 for evaluation.
