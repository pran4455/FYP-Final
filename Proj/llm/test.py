import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline,
)
from peft import LoraConfig, get_peft_model


# ---------------------------
# CONFIG
# ---------------------------
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
DATASET_PATH = "Driver_Stress_Instruction_Dataset.jsonl"
OUTPUT_DIR = "./driver_stress_chatbot_lora"


# ---------------------------
# CHECK GPU
# ---------------------------
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ---------------------------
# LOAD DATASET
# ---------------------------
dataset = load_dataset("json", data_files=DATASET_PATH)["train"]

def format_prompt(example):
    return {
        "text": f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    }

original_columns = dataset.column_names
dataset = dataset.map(format_prompt)
dataset = dataset.remove_columns(original_columns)

print("Rows:", len(dataset))
print(dataset[0]["text"])


# ---------------------------
# TOKENIZER
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)


# ---------------------------
# 4-BIT MODEL LOAD
# ---------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    low_cpu_mem_usage=True
)


# ---------------------------
# APPLY LORA
# ---------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["qkv_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

model.gradient_checkpointing_enable()
model.print_trainable_parameters()


# ---------------------------
# TRAINING ARGUMENTS
# ---------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=200,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    save_total_limit=2
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=data_collator
)


# ---------------------------
# TRAIN
# ---------------------------
trainer.train()


# ---------------------------
# SAVE MODEL
# ---------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Model saved to", OUTPUT_DIR)


# ---------------------------
# LOAD FOR INFERENCE
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    device_map="auto",
    low_cpu_mem_usage=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)


# ---------------------------
# TEST PROMPT
# ---------------------------
test_prompt = """### Instruction:
Generate a calm, safety-focused response based on the driver's current state.

### Input:
stress_level: high
stress_score: 82
context: heavy traffic
required_response_style: calming
user_report: I feel very tense and stuck in traffic

### Response:"""

output = pipe(test_prompt, max_new_tokens=120, do_sample=True, temperature=0.7)
print("\n===== MODEL OUTPUT =====\n")
print(output[0]["generated_text"])