"""
Download language models from HuggingFace.

Detailed description:
    This module provides functionality for downloading language models from 
    HuggingFace to specified local directories.

Usage example:
    >>> module load python3/3.10.4
    >>> source /scratch/wd04/bk2508/venvs/llm-env/bin/activate
    >>> export HF_TOKEN=
    >>> python3 download.py

Author:
    Binod Karunanayake

Created:
    2026-01-02
"""

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import hub
import os

hub.HF_MODULES_CACHE="/scratch/wd04/bk2508/tmp"
hub.PYTORCH_PRETRAINED_BERT_CACHE="/scratch/wd04/bk2508/tmp"
hub.PYTORCH_TRANSFORMERS_CACHE="/scratch/wd04/bk2508/tmp"
hub.TRANSFORMERS_CACHE="/scratch/wd04/bk2508/tmp"

model_name = input("Model name: ").strip()

try:
    snapshot_download(
        repo_id=model_name,
        local_dir=f"/scratch/wd04/bk2508/models/{model_name}",
        token=os.getenv("HF_TOKEN")
    )
except Exception as e:
    print(f"Error downloading {model_name}: {e}")
