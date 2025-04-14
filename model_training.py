from transformers import AutoTokenizer, AutoModelForCausalLM

model = 'meta-llama/Llama-3.1-8B-Instruct'

import accelerate
from accelerate import *

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(
    model, device_map="auto"
)

tag_msg = """<button aria-label="Mobile menu" class="mega-toggle-animated mega-toggle-animated-slider" type="button" aria-expanded="false">
                  <span class="mega-toggle-animated-box">
                    <span class="mega-toggle-animated-inner"></span>
                  </span>
                </button>
"""

messages = [
    {
        "role": "system",
        "content": "You are a very proficient HTML developer and you can tell what a tag is for just by looking at it",
    },
    {"role": "user", "content": f"What is this tag for: {tag_msg}"},
]
model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
input_length = model_inputs.shape[1]
generated_ids = model.generate(model_inputs, do_sample=True, max_new_tokens=2000)
print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])