import json

INPUT_FILE = "Driver_Stress_Dataset.jsonl"
OUTPUT_FILE = "Driver_Stress_Instruction_Dataset.jsonl"

INSTRUCTION = (
    "Generate a calm, safety-focused response based on the driver's current state. "
    "Acknowledge the risk briefly and suggest concrete actions to reduce stress and improve safety."
)

def build_input(sample):
    lines = []

    if "stress_level" in sample:
        lines.append(f"stress_level: {sample['stress_level']}")

    if "stress_score" in sample:
        lines.append(f"stress_score: {sample['stress_score']}")

    if "category" in sample:
        lines.append(f"context: {sample['category']}")

    # Optional signal derived from response_type
    if sample.get("response_type"):
        lines.append(f"required_response_style: {sample['response_type']}")

    # Original user message (if useful)
    user_msg = next(
        (m["content"] for m in sample["messages"] if m["role"] == "user"),
        None
    )
    if user_msg:
        lines.append(f"user_report: {user_msg}")

    return "\n".join(lines)


with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for line in infile:
        sample = json.loads(line)

        assistant_msg = next(
            (m["content"] for m in sample["messages"] if m["role"] == "assistant"),
            None
        )

        if not assistant_msg:
            continue

        transformed = {
            "instruction": INSTRUCTION,
            "input": build_input(sample),
            "output": assistant_msg
        }

        outfile.write(json.dumps(transformed, ensure_ascii=False) + "\n")

print(f"Converted dataset saved as: {OUTPUT_FILE}")
