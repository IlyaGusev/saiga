import copy
import random
from typing import List, Dict

from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def fix_messages(messages):
    messages = copy.deepcopy(messages)
    for m in messages:
        m["content"] = [{"type": "text", "text": m["content"]}]
    return messages


class DPODataset(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        tokenizer: AutoTokenizer,
        max_tokens_count: int,
        sample_rate: float = 1.0,
        apply_chat_template: bool = False,
    ):
        self.original_records = original_records
        self.tokenizer = tokenizer
        self.max_tokens_count = max_tokens_count
        self.sample_rate = sample_rate

        self.records = []
        for record in tqdm(original_records):
            if random.random() > self.sample_rate:
                continue
            prompt_messages = fix_messages(record["prompt"])
            chosen_messages = fix_messages(record["chosen"])
            rejected_messages = fix_messages(record["rejected"])

            chosen_tokens = self.tokenizer.apply_chat_template(
                prompt_messages + chosen_messages,
                add_generation_prompt=False,
                tokenize=True,
            )
            if isinstance(chosen_tokens[0], list):
                chosen_tokens = chosen_tokens[0]

            rejected_tokens = self.tokenizer.apply_chat_template(
                prompt_messages + rejected_messages,
                add_generation_prompt=False,
                tokenize=True,
            )
            if isinstance(rejected_tokens[0], list):
                rejected_tokens = rejected_tokens[0]

            if len(chosen_tokens) > self.max_tokens_count - 5:
                continue
            if len(rejected_tokens) > self.max_tokens_count - 5:
                continue
            if not apply_chat_template:
                self.records.append(
                    {
                        "prompt": prompt_messages,
                        "chosen": chosen_messages,
                        "rejected": rejected_messages,
                    }
                )
            else:
                prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False)

                chosen = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
                chosen = chosen.replace(tokenizer.bos_token, "")

                rejected = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
                rejected = rejected.replace(tokenizer.bos_token, "")

                assert chosen.strip()
                assert rejected.strip()
                self.records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]
