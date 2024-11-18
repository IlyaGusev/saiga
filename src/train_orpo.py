import json
import random
import fire
from typing import List, Dict

import wandb
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from trl import ORPOConfig, ORPOTrainer
from datasets import Dataset as HFDataset
from unsloth import PatchDPOTrainer, FastLanguageModel

from src.dpo_dataset import DPODataset
from src.util.io import read_jsonl


def train(
    config_file: str,
    train_path: str,
    eval_path: str,
    output_dir: str,
    sample_rate: float = 1.0,
):
    PatchDPOTrainer()
    with open(config_file, "r") as r:
        config = json.load(r)

    max_tokens_count = config["max_tokens_count"]
    max_seq_length = config.get("max_seq_length", max_tokens_count)
    model_name = config["model_name"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_8bit=config["load_in_8bit"],
        load_in_4bit=config["load_in_4bit"],
        attn_implementation="flash_attention_2",
    )
    tokenizer.pad_token = config["pad_token"]
    tokenizer.eos_token = config["eos_token"]
    tokenizer.bos_token = config["bos_token"]
    tokenizer.padding_side = "left"
    tokenizer.save_pretrained(output_dir)

    lora_config = config["lora"]
    if lora_config:
        model = FastLanguageModel.get_peft_model(
            model, **config["lora"], max_seq_length=max_seq_length
        )

    train_records = read_jsonl(train_path)
    train_dataset = DPODataset(
        train_records,
        tokenizer=tokenizer,
        max_tokens_count=max_tokens_count,
        sample_rate=sample_rate,
    )
    train_dataset = HFDataset.from_list(train_dataset)
    eval_records = read_jsonl(eval_path)
    eval_dataset = DPODataset(
        eval_records,
        tokenizer=tokenizer,
        max_tokens_count=max_tokens_count,
        sample_rate=sample_rate,
    )
    eval_dataset = HFDataset.from_list(eval_dataset)
    print(train_dataset[0])

    trainer_config = config.get("trainer")
    if trainer_config.get("report_to", "wandb") == "wandb":
        wandb.init(project="rulm_self_instruct", name=config_file)

    training_args = ORPOConfig(
        output_dir=output_dir, report_to="wandb", **config["orpo"], **trainer_config
    )

    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
