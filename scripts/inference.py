from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

# Load tokenizer and model (adjust path to local directory or model hub name)
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Define system and user prompts (ChatML format)
def format_chat(system_prompt, user_prompt):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>" \
           f"<|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|>" \
           f"<|start_header_id|>assistant<|end_header_id|>\n"

# Example usage
system_prompt = "You are a helpful assistant."
user_prompt = "What's the capital of France?"
input_text = format_chat(system_prompt, user_prompt)

# Tokenize and generate
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
_ = model.generate(**inputs, streamer=streamer, max_new_tokens=100)
