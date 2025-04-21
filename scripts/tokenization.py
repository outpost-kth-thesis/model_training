from transformers import AutoTokenizer
from dotenv import load_dotenv
import os
import random

load_dotenv()

_system_prompts = [
    "You are an expert Javascript coder who is really good at deciphering minified Javascript code, but you only only in plain Javascript with no added text"
    "You are an expert in reading minified JavaScript, but you stick to pure JavaScript without adding explanations or extra context",
    "You really know your way around minified JavaScript, but you only write in plain JavaScript—no added notes, no extra chatter",
    "Act as a JavaScript expert who can expertly read minified code, but respond only using plain JavaScript, without any added context or commentary."
]

def tokenize(input, output, max_length=1028, truncation=True, padding="max_length"):
    formatted_prompt = format_llama3(input=input)
    tokenizer = get_tokenizer()
    return tokenizer(formatted_prompt, text_target=output, padding=padding, max_length=max_length, truncation=truncation, return_tensors="pt")

def get_tokenizer():
    _tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_NAME"))
    special_tokens = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>"
    ]
    _tokenizer.add_special_tokens({
        "additional_special_tokens": special_tokens
    })

    _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer

pad_token = get_tokenizer().eos_token
    
def format_llama3(input):
    return f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{random.choice(_system_prompts)}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Deminify this code sample for me: 
{input}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
""".strip()


if __name__ == "__main__":
    input = """console.log("hello world")
console.log("more text")
    """

    output = """console.log("hello world")
console.log("more text")
    """

    print(format_llama3(input))
    print(tokenize(input, output))