"""
Fine-tune language models with dp using unsloth & opacus.

Detailed description:
    This module provides functionality to fine-tune language models using the 
    unsloth library. The dataset should be downloaded before running this script.

    ```python
    from datasets import load_dataset
    dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
    dataset.save_to_disk("FineTome-100k-gemma-3-4b-it-train")
    ```

Usage example:
    >>> qsub jobscript.sh

Author:
    Binod Karunanayake

Created:
    2026-01-28
"""

from unsloth import FastModel
import torch
import sys
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only, get_chat_template, standardize_data_formats
from datasets import load_dataset, Dataset, load_from_disk
import pandas as pd
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# Set job id and model name
job_id = sys.argv[1].split(".")[0]
model_name = sys.argv[2]
dataset_name = sys.argv[3]
chat_template_name = "gemma-3"

# Differential Privacy Configuration
enable_dp = True  # Set to True to enable differential privacy
target_epsilon = 8.0  # Privacy budget (lower = stronger privacy)
target_delta = 1e-5  # Delta parameter (typically 1/number_of_samples)
max_grad_norm = 1.0  # Gradient clipping threshold

# Set output directory
output_dir = f"/scratch/wd04/bk2508/repositories/iot-llm-grpo/fine-tuned-models/{dataset_name}-{model_name.split('/')[1].lower()}-{job_id}"

model, tokenizer = FastModel.from_pretrained(
    model_name = f"/scratch/wd04/bk2508/models/{model_name}",
    local_files_only=True,
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

# Validate model for differential privacy if enabled
if enable_dp:
    model = ModuleValidator.fix(model)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = chat_template_name,
)

def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

dataset = load_from_disk(f"/scratch/wd04/bk2508/repositories/iot-llm-grpo/data/{dataset_name}-{model_name.split('/')[1].lower()}-train")
dataset = standardize_data_formats(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True)

# Adjust batch size for differential privacy to account for noise
if enable_dp:
    # DP-SGD requires careful tuning of batch size and learning rate
    per_device_batch_size = 2
    gradient_accumulation = 4
    learning_rate = 1e-4  # Reduce learning rate for DP training
else:
    per_device_batch_size = 2
    gradient_accumulation = 4
    learning_rate = 2e-4

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field = "text",
        per_device_train_batch_size = per_device_batch_size,
        gradient_accumulation_steps = gradient_accumulation, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = learning_rate,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

# Attach privacy engine to trainer if differential privacy is enabled
if enable_dp:
    print("Enabling Differential Privacy...")
    privacy_engine = PrivacyEngine()
    trainer.model, trainer.optimizer, trainer.train_dataloader = privacy_engine.make_private(
        module=trainer.model,
        optimizer=trainer.optimizer,
        data_loader=trainer.train_dataloader,
        noise_multiplier=1.0,  # Controls privacy-utility tradeoff
        max_grad_norm=max_grad_norm,
    )
    print(f"Privacy Engine attached with epsilon={target_epsilon}, delta={target_delta}")

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Print differential privacy information if enabled
if enable_dp:
    epsilon = privacy_engine.accountant.get_epsilon(target_delta)
    print(f"\n--- Differential Privacy Summary ---")
    print(f"Differential Privacy Enabled: {enable_dp}")
    print(f"Target Delta: {target_delta}")
    print(f"Achieved Epsilon: {epsilon:.4f}")
    print(f"Max Gradient Norm: {max_grad_norm}")
