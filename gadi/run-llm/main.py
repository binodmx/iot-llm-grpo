"""
Run language models using Transformers.

Detailed description:
    This module provides functionality to run language models using the 
    Transformers library with 4-bit quantization for memory efficiency.

Usage example:
    >>> qsub jobscript.sh

Author:
    Binod Karunanayake

Created:
    2026-01-02
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import sys

# Set job id and model name
job_id = sys.argv[1].split(".")[0]
model_name = sys.argv[2]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    f"/scratch/wd04/bk2508/models/{model_name}",
    local_files_only=True
)

# Set pad_token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4-bit quantization config
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    f"/scratch/wd04/bk2508/models/{model_name}",
    device_map="auto",
    local_files_only=True,
    quantization_config=config,
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# Input prompt
input_text = "Why is the sky blue?"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Generate text
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.pad_token_id,
        max_length=150,         # Maximum length of the output
        num_return_sequences=1, # Number of responses
        do_sample=True,
        temperature=0.7,        # Adjust creativity level
        top_p=0.9,              # Nucleus sampling
    )

# Decode and print the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Device: {model.device}")
print(f"Output: {generated_text}")

