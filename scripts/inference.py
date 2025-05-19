import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Load model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
use_4bit = True

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",     # NormalFloat4
    bnb_4bit_compute_dtype=torch.float16,
    load_in_8bit=not use_4bit
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map="auto"
)

# Prompt formatting (ChatML-style)
def format_prompt(system_prompt, user_prompt):
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + system_prompt + "<|eot_id|>" +
        "<|start_header_id|>user<|end_header_id|>\n" + user_prompt + "<|eot_id|>" +
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

# Generate response
def chat(system_prompt, user_prompt):
    prompt = format_prompt(system_prompt, user_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    eos_token_id = tokenizer.eos_token_id
    output = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=eos_token_id,
        max_new_tokens=4096  # high enough to avoid artificial cutoff
    )

    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Get the text after the assistant prompt
    assistant_start = output_text.find(user_prompt) + len(user_prompt)
    assistant_response = output_text[assistant_start:].strip()

    print("\n--- Generated Text ---")
    print(assistant_response)
    print("----------------------\n")
    return assistant_response


system_prompt = """
You are really good at filling out forms online. You are given the input as a list, you return the output as JSON.
When given a list of HTML elements, you are able provide an appropriate input for each of them. If you cannot, you set the status for the tag to be "NEED_INFO".
Whenever you are filling a form, you use the following personal information:
Your full name is John Doe.
Your email address is john.doe@gmail.com.
Your password is qwertyuiopzxcvbnm.
Your mobile number is 0123456789.
If the input tag is a text field, then you give an appropriate value for it. 
If the input tag is a button, you say inp_click.
If the input tag is a checkbox, you say how many options to check

You also only answer in JSON using the following structure:
[
    {
        "verbatim_tag": the tag verbatim,
        "input_type": button | textfield | checkbox,
        "status": NEED_INFO | OK
        "value": inp_click | text value | select option
    }
]
"""
user_prompt = """
---
Element HTML:
<input type="text" name="first_name" id="first_name" class="block py-2.5 px-0 w-full text-sm text-gray-900 bg-transparent border-0 border-b-2 border-gray-300 appearance-none focus:outline-none focus:ring-0 focus:border-blue-600 peer" placeholder=" " required="">
---
Element HTML:
<input type="text" name="last_name" id="last_name" class="block py-2.5 px-0 w-full text-sm text-gray-900 bg-transparent border-0 border-b-2 border-gray-300 appearance-none focus:outline-none focus:ring-0 focus:border-blue-600 peer" placeholder=" " required="">
---
Element HTML:
<input type="text" name="company" id="company" class="block py-2.5 px-0 w-full text-sm text-gray-900 bg-transparent border-0 border-b-2 border-gray-300 appearance-none focus:outline-none focus:ring-0 focus:border-blue-600 peer" placeholder=" " required="">
---
Element HTML:
<button type="submit" class="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm w-full sm:w-auto px-5 py-2.5 text-center dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800">Submit</button>
"""
# Example usage
if __name__ == "__main__":
    chat(system_prompt, user_prompt)