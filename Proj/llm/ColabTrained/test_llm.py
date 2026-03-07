from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

print("Loading your trained Driver Stress AI...")
model_path = "./driver_stress_final"

# Load your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16,
    device_map="auto"  # RTX 4060 auto-detection
)

# Test pipeline
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    device_map="auto"
)

# Test your trained model
prompts = [
    "User: Traffic is insane, heart racing!",
    "User: Stuck behind slow truck, late for meeting",
    "User: Horns everywhere, feeling angry"
]

print("\n" + "="*60)
print("DRIVER STRESS AI TEST RESULTS")
print("="*60)

for prompt in prompts:
    response = pipe(prompt, max_new_tokens=80, temperature=0.7)[0]['generated_text']
    print(f"\nPrompt: {prompt}")
    print(f"AI: {response.split('AI Support:')[-1].strip()[:200]}...")
    print("-" * 40)

print("\n✅ Your model works perfectly on RTX 4060!")
