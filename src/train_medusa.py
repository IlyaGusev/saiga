import os
import json
from collections import Counter
from typing import Optional, List

import types
import unsloth
from unsloth import FastLanguageModel, UnslothTrainingArguments
from transformers import DataCollatorForTokenClassification, Trainer
import fire
import wandb
import torch

from src.dataset import ChatDataset
from src.util.io import read_jsonl
from safetensors.torch import save_file

os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._dynamo.config.cache_size_limit = 128


class ResBlock(torch.nn.Module):
    def __init__(self, hidden_size, dtype=torch.bfloat16):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size, dtype=dtype)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


def add_medusa_heads(
    model: FastLanguageModel,
    medusa_num_heads: int = 3,
    medusa_num_layers: int = 1,
    dtype=torch.bfloat16,
):
    lm_head = model.lm_head
    hidden_size = lm_head.weight.shape[-1]
    vocab_size = lm_head.weight.shape[0]
    model.config.medusa_num_layers = medusa_num_layers
    model.config.medusa_num_heads = medusa_num_heads
    model.medusa_num_heads = medusa_num_heads

    model.medusa_head = torch.nn.ModuleList(
        [
            torch.nn.Sequential(
                *([ResBlock(hidden_size, dtype=dtype)] * medusa_num_layers),
                torch.nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype),
            )
            for _ in range(medusa_num_heads)
        ]
    )
    model.medusa_head.to(model.dtype).to(model.device)
    for i in range(medusa_num_heads):
        model.medusa_head[i][-1].weight.data[:] = model.lm_head.weight.data[:]
    model.medusa_head.to(model.dtype).to(model.device)

    model.old_forward = model.forward

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            medusa_logits = [self.lm_head(hidden_states)]
        for i in range(self.medusa_num_heads):
            medusa_logits.append(self.medusa_head[i](hidden_states))
        return torch.stack(medusa_logits, dim=0)

    model.forward = types.MethodType(forward, model)


class CustomTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        logits = model(**inputs)
        labels = inputs["labels"]
        loss = 0
        log = dict()
        loss_fct = torch.nn.CrossEntropyLoss()
        medusa = logits.shape[0]
        for i in range(medusa):
            medusa_logits = logits[i, :, : -(1 + i)].contiguous()
            medusa_logits = medusa_logits.view(-1, logits.shape[-1])

            medusa_labels = labels[..., 1 + i :].contiguous()
            medusa_labels = medusa_labels.view(-1)
            medusa_labels = medusa_labels.to(medusa_logits.device)
            loss_i = loss_fct(medusa_logits, medusa_labels)
            loss += loss_i

            not_ignore = medusa_labels.ne(-100)
            medusa_labels = medusa_labels[not_ignore]

            _, topk = medusa_logits.topk(1, dim=-1)
            topk = topk[not_ignore]
            correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
            log[f"medusa{i}_top1"] = correct.float().mean().item()
            log[f"medusa{i}_loss"] = loss_i.item()
        prefix = "train" if model.training else "eval"
        log = {f"{prefix}/{k}": v for k, v in log.items()}
        wandb.log(
            {
                **log,
                "train/global_step": self.state.global_step,
            }
        )
        return (loss, logits) if return_outputs else loss


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
        attn_implementation="flash_attention_2",
    )
    for param in model.parameters():
        param.requires_grad = False
    tokenizer.save_pretrained(output_dir)

    medusa_config = config["medusa"]
    add_medusa_heads(
        model, medusa_config["medusa_num_heads"], medusa_config["medusa_num_layers"]
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
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, pad_to_multiple_of=8
    )

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
    medusa_head = model.medusa_head
    state_dict = medusa_head.state_dict()
    save_file(
        state_dict,
        os.path.join(output_dir, "medusa_head.safetensors"),
    )


if __name__ == "__main__":
    fire.Fire(train)
