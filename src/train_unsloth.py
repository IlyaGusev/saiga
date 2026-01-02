import os
import json
from collections import Counter

import unsloth
from unsloth import FastLanguageModel, UnslothTrainingArguments
from unsloth.trainer import _create_unsloth_optimizer
from transformers import DataCollatorForTokenClassification, Trainer
import fire
import wandb
import torch

from src.dataset import ChatDataset
from src.util.io import read_jsonl

os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._dynamo.config.cache_size_limit = 128


class CustomTrainer(Trainer):
    def create_optimizer(self):
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None:
            return super().create_optimizer()
        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = _create_unsloth_optimizer(
                self.model,
                optimizer_cls,
                optimizer_kwargs,
                embedding_learning_rate,
            )
        return self.optimizer


@torch.inference_mode
def fix_untrained_tokens(model, tokenizer, eps: float = 1e-16) -> None:
    embedding_matrix = model.get_input_embeddings().weight
    lm_head_matrix = model.get_output_embeddings().weight
    assert embedding_matrix.shape[0] == lm_head_matrix.shape[0]

    indicator_untrained = torch.amax(embedding_matrix, axis=1) <= eps
    special_tokens = (
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    )
    for special_token in special_tokens:
        if hasattr(tokenizer, special_token + "_id"):
            token_id = eval(f"tokenizer.{special_token}_id")
            if token_id is not None and token_id < indicator_untrained.shape[0]:
                indicator_untrained[token_id] = False

    where_untrained = torch.where(indicator_untrained)[0]
    n_untrained = where_untrained.shape[0]
    n_trained = embedding_matrix.shape[0] - n_untrained
    if n_untrained == 0:
        return

    where_untrained = where_untrained.tolist()
    actual_bad_tokens = tokenizer.convert_ids_to_tokens(where_untrained)
    actual_bad_tokens = [x for x in actual_bad_tokens if x is not None]

    sum_embedding = torch.sum(embedding_matrix, dtype=torch.float32, axis=0)
    sum_lm_head = torch.sum(lm_head_matrix, dtype=torch.float32, axis=0)

    sum_embedding -= torch.sum(embedding_matrix[where_untrained], dtype=torch.float32, axis=0)
    sum_lm_head -= torch.sum(lm_head_matrix[where_untrained], dtype=torch.float32, axis=0)

    mean_embedding = sum_embedding / n_trained
    mean_lm_head = sum_lm_head / n_trained

    mean_embedding = mean_embedding.repeat((n_untrained, 1))
    mean_lm_head = mean_lm_head.repeat((n_untrained, 1))

    embedding_matrix[where_untrained] = mean_embedding.to(embedding_matrix.dtype)
    lm_head_matrix[where_untrained] = mean_lm_head.to(lm_head_matrix.dtype)

    torch.cuda.empty_cache()


def train(
    config_path: str,
    train_path: str,
    val_path: str,
    output_dir: str,
    sample_rate: float = 1.0,
) -> None:
    with open(config_path) as r:
        config = json.load(r)

    max_tokens_count = config["max_tokens_count"]
    max_seq_length = config.get("max_seq_length", max_tokens_count)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_8bit=config["load_in_8bit"],
        load_in_4bit=config["load_in_4bit"],
        full_finetuning=(config.get("lora") is None),
    )
    tie_word_embeddings = model.config.tie_word_embeddings
    tokenizer.bos_token = config["bos_token"]
    tokenizer.eos_token = config["eos_token"]
    tokenizer.pad_token = config["pad_token"]
    tokenizer.padding_side = "left"
    tokenizer.save_pretrained(output_dir)

    if config.get("fix_untrained_tokens", False):
        fix_untrained_tokens(model, tokenizer)

    lora_config = config.get("lora")
    if lora_config:
        model = FastLanguageModel.get_peft_model(
            model,
            **config["lora"],
            max_seq_length=max_seq_length,
        )
        modules_to_save = config["lora"].get("modules_to_save", [])
        if (
            tie_word_embeddings
            and "embed_tokens" in modules_to_save
            and "lm_head" in modules_to_save
            and "gemma3" not in config["model_name"]
            and "gemma-3" not in config["model_name"]
        ):
            print("Tying lm_head and embed_tokens...")
            model.base_model.model.model.embed_tokens.modules_to_save["default"].weight = (
                model.base_model.model.lm_head.modules_to_save["default"].weight
            )

    train_records = read_jsonl(train_path)
    val_records = read_jsonl(val_path)

    datasets = []
    for records in (train_records, val_records):
        datasets.append(
            ChatDataset(
                records,
                tokenizer,
                max_tokens_count=max_tokens_count,
                sample_rate=sample_rate,
                only_target_loss=config["only_target_loss"],
                add_global_bos=config.get("add_global_bos", True),
                add_global_eos=config.get("add_global_eos", True),
            )
        )
    train_dataset, val_dataset = datasets
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer_config = config["trainer"]
    if trainer_config.get("report_to", "wandb") == "wandb":
        wandb.init(project="rulm_self_instruct", name=config_path)
    trainer = CustomTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        args=UnslothTrainingArguments(**trainer_config, output_dir=output_dir),
    )
    trainer.train()
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
