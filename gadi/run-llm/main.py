"""
Run language models using Transformers.

Detailed description:
    This module provides functionality to run language models using the 
    unsloth library with quantization for memory efficiency.

Usage example:
    >>> qsub jobscript.sh

Author:
    Binod Karunanayake

Created:
    2026-01-02
"""

from unsloth import FastModel
import torch
import sys

# Set job id and model name
job_id = sys.argv[1].split(".")[0]
model_name = sys.argv[2]

model, tokenizer = FastModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
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

