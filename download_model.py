import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

base_save_dir = "/data/models"
model_name_for_path = model_name.replace("/", "--")
save_path = os.path.join(base_save_dir, model_name_for_path)

os.makedirs(save_path, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model saved to: {save_path}")
