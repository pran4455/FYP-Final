from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import torch

print(f"RTX 4060 Laptop GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
print(f"CUDA: {torch.cuda.is_available()}")

# 1. Load model (RTX 4060 optimized)
print("\nLoading DeepSeek-R1 1.5B (3.2GB)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-1.5b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    trust_remote_code=True,
)

# 2. Load YOUR dataset
print("\nLoading 500-example dataset")
dataset = load_dataset("json", data_files="Driver_Stress_500_Dataset.jsonl", split="train")
print(f"Dataset loaded: {len(dataset)} examples")

# 3. FIXED formatting function (your bug fixed)
def formatting_prompts_func(examples):
    texts = []
    for i in range(len(examples["messages"])):
        user_msg = examples["messages"][i][0]["content"]
        assistant_msg = examples["messages"][i][1]["content"]
        text = f"### User:\n{user_msg}\n\n### Assistant:\n{assistant_msg}<|im_end|>"
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
print("Dataset formatted")

# 4. RTX 4060 LoRA config
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
print("LoRA adapters applied")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,        
        gradient_accumulation_steps=4,         
        warmup_steps=10,
        num_train_epochs=3,                    
        learning_rate=2e-4,
        fp16=True,                            
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="driver_stress_logs",
        save_strategy="epoch",
        save_total_limit=2,
    ),
)

print("\nStarting training")
trainer.train()

model.save_pretrained("driver_stress_final")
tokenizer.save_pretrained("driver_stress_final")
print("\nTRAINING COMPLETE! Model saved: driver_stress_final/")
