from transformers import AutoTokenizer
from dotenv import load_dotenv
import os
import random

load_dotenv()

_tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_NAME"))
_tokenizer.pad_token = _tokenizer.eos_token
_system_prompts = [
    "You are an expert Javascript coder who is really good at deciphering minified Javascript code, but you only only in plain Javascript with no added text"
    "You’re an expert in reading minified JavaScript, but you stick to pure JavaScript without adding explanations or extra context",
    "You really know your way around minified JavaScript, but you only write in plain JavaScript—no added notes, no extra chatter",
    "Act as a JavaScript expert who can expertly read minified code, but respond only using plain JavaScript, without any added context or commentary."
]

def tokenize(input, output, max_length=1028, truncation=False, padding="longest"):
    formatted_prompt = format_llama3(input=input, output=output)
    return _tokenizer(formatted_prompt, padding=padding, max_length=max_length, truncation=truncation)
    
def format_llama3(input, output):
    return f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{random.choice(_system_prompts)}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Deminify this code sample for me: 
{input}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{output}
<|eot_id|>
""".strip()


if __name__ == "__main__":
    input = """console.log("hello world")
console.log("more text")
    """

    output = """console.log("hello world")
console.log("more text")
    """

    print(format_llama3(input, output))
    print(tokenize(input, output))