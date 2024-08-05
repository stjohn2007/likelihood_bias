import os

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_llama2(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if "HUGGINGFACE_HUB_CACHE" in os.environ:
        cache_dir = os.environ["HUGGINGFACE_HUB_CACHE"]
    else:
        cache_dir = None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=False,
        cache_dir=cache_dir,
    )
    return model, tokenizer
