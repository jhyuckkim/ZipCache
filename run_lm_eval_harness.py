import torch
from transformers import AutoTokenizer
import lm_eval
import os
import json

from zipcache import MyLlamaForCausalLM

# Define compress_config
compress_config = {}
compress_config["compress_mode"] = "mixed_channelwiseQ"
compress_config["quantize_bit_important"] = 4
compress_config["quantize_bit_unimportant"] = 2
compress_config["k_unimportant_ratio"] = 0.4
# Value compress config
compress_config["v_compress_mode"] = "channel_separate_mixed_tokenwiseQ"
compress_config["v_quantize_bit_important"] = 4
compress_config["v_quantize_bit_unimportant"] = 2
compress_config["v_unimportant_ratio"] = 0.4
compress_config["stream"] = True  # streaming-gear set to true to perform better efficiency
compress_config["streaming_gap"] = 100  # re-compress every N iteration

MODEL_PATH = '/data/models/meta-llama--Meta-Llama-3-8B-Instruct'
SAVE_PATH = '/lm_eval_results'
results_file_name = "Meta-Llama-3-8B-Instruct.json"  # Replace with your desired file name

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, use_fast=True, cache_dir=MODEL_PATH, local_files_only=True
)

if 'Llama' in MODEL_PATH:
    model = MyLlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        cache_dir=MODEL_PATH,
        compress_config=compress_config,
        torch_dtype=torch.float16,
        local_files_only=True
    )
else:
    raise NotImplementedError

model.half().eval().cuda()

# Corrected variable name from 'custom_model' to 'model'
lm_obj = lm_eval.models.huggingface.HFLM(
    pretrained=model,  # Pass your custom model instance
    tokenizer=tokenizer,
    device="cuda",
)

task_manager = lm_eval.tasks.TaskManager()
tasks = ["gsm8k"]

task_dict = lm_eval.tasks.get_task_dict(
    tasks,
    task_manager
)

print("task_dict", task_dict)

results = lm_eval.evaluate(
    lm=lm_obj,
    task_dict=task_dict,
)

print(results["results"])

# Save results if the path is specified
if SAVE_PATH:
    os.makedirs(SAVE_PATH, exist_ok=True)
    results_file = os.path.join(SAVE_PATH, results_file_name)
    with open(results_file, 'w') as f:
        json.dump(results, f)
    print(f"Results saved to {results_file}")
