from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Hugging Face model
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model loaded successfully!")

# User question
question = "What is an ultrasonic sensor?"

# Chat-style prompt
messages = [
    {
        "role": "system",
        "content": "You are a helpful technical assistant. Give clear and concise answers."
    },
    {
        "role": "user",
        "content": question
    }
]

# Convert chat messages into the format expected by Qwen
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Convert text into tokens
inputs = tokenizer(text, return_tensors="pt")

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150
    )

# Only decode the newly generated tokens
generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(answer)