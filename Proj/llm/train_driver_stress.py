import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import torch

print("🚗 DRIVER STRESS AI - RTX 4060 LAPTOP TRAINING")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
print("-" * 60)

# 1. Load model
print("📥 Loading DeepSeek-R1 1.5B...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-1.5b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    trust_remote_code=True,
)

# 2. Load dataset
print("📚 Loading 500 examples...")
dataset = load_dataset("json", data_files="Driver_Stress_500_Dataset.jsonl", split="train")
print(f"Loaded: {len(dataset)} conversations")

# 3. Format dataset (CORRECTED)
def format_for_training(examples):
    texts = []
    for messages in examples["messages"]:
        user_input = messages[0]["content"]
        ai_response = messages[1]["content"]
        text = f"User: {user_input}\n\nAssistant: {ai_response}<|im_end|>"
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_for_training, batched=True)
print("✅ Dataset formatted")

# 4. LoRA setup
model = FastLanguageModel.get_peft_model(
    model, 
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print("✅ LoRA ready")

# 5. RTX 4060 optimized training
training_args = TrainingArguments(
    per_device_train_batch_size=2,      # Safe for laptop
    gradient_accumulation_steps=4,      # Effective batch=8
    warmup_steps=10,
    max_steps=375,                      # ~3 epochs
    learning_rate=2e-4,
    fp16=True,
    logging_steps=25,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="driver_stress_final",
    save_strategy="steps",
    save_steps=125,
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=0,           # Windows stable
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=training_args,
)

print("\n🚀 STARTING TRAINING (90 mins expected)")
print("💾 Watch VRAM: nvidia-smi")
trainer.train()

# 6. Save
model.save_pretrained("driver_stress_final")
tokenizer.save_pretrained("driver_stress_final")
print("\n🎉 SUCCESS! Model saved: driver_stress_final/")
print("Next: ollama create driver_stress_ai -f Modelfile")
