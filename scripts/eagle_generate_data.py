import os

from tqdm import tqdm
import fire
import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from src.dataset import ChatDataset
from src.util.io import read_jsonl


def eagle_generate_data(
    input_path: str,
    out_dir: str,
    model_name: str,
    max_tokens_count: int = 4096,
    sample_rate: float = 1.0,
    group_size: int = 5000,
):
    records = read_jsonl(input_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_tokens_count,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
        attn_implementation="flash_attention_2",
    )
    dataset = ChatDataset(
        records,
        tokenizer,
        sample_rate=sample_rate,
        max_tokens_count=max_tokens_count,
        only_target_loss=True,
        add_global_bos=True,
        add_global_eos=True,
    )
    for idx, row in tqdm(enumerate(dataset), total=len(dataset)):
        group_size = group_size
        start = (idx // group_size) * group_size
        end = start + group_size
        grouped_subdir = f"rows_{start}-{end}"
        if not os.path.exists(f"{out_dir}/{grouped_subdir}"):
            os.makedirs(f"{out_dir}/{grouped_subdir}")
        output_file = f"{out_dir}/{grouped_subdir}/data_{idx}.ckpt"
        if os.path.exists(output_file):
            continue
        with torch.no_grad():
            outputs = model(
                row["input_ids"].unsqueeze(0).cuda(), output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1].cpu()
        loss_mask = row["labels"] != -100
        data_point = {
            "conversation_str": tokenizer.decode(
                row["input_ids"], skip_special_tokens=False
            ),
            "input_ids": row["input_ids"],
            "loss_mask": loss_mask,
            "hidden_state": hidden_states,
        }
        torch.save(data_point, output_file)


if __name__ == "__main__":
    fire.Fire(eagle_generate_data)
