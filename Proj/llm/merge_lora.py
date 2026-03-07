import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "microsoft/phi-3-mini-4k-instruct"
lora_path = "./driver_stress_chatbot_lora"
output_dir = "./merged_model"

# Load tokenizer from base model
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map=None
)

# Load LoRA
model = PeftModel.from_pretrained(model, lora_path)

# Merge LoRA
model = model.merge_and_unload()

# Save merged model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("✅ Merge complete")
