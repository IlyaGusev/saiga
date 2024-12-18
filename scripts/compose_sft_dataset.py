import json
import copy
import random
from itertools import chain

import mmh3
import fire
from datasets import load_dataset


def compose_sft_dataset(config_path: str, train_path: str, val_path: str):
    with open(config_path) as r:
        config = json.load(r)

    records = []
    dataset_name = config.get("dataset_name", "IlyaGusev/saiga_scored")
    revision = config["dataset_revision"]
    add_rejected = config.get("add_rejected", False)
    if isinstance(dataset_name, str):
        dataset = load_dataset(dataset_name, split="train", revision=revision)
    elif isinstance(dataset_name, list):
        dataset = chain(*[load_dataset(name, split="train", revision=r) for name, r in zip(dataset_name, revision)])

    for row in dataset:
        is_bad_by_regex = row.get("is_bad_by_regex", False)
        if config.get("exclude_regex", False) and is_bad_by_regex:
            continue

        score = row.get("opus_score")
        if score is not None and score < config.get("min_score", 8):
            continue

        all_messages = []
        if "messages" in row:
            all_messages.append(row["messages"])
        elif "chosen" in row:
            all_messages.append(row["prompt"] + row["chosen"])
            if add_rejected:
                all_messages.append(row["prompt"] + row["rejected"])
        else:
            assert False

        mapping = {"bot": "assistant"}
        for messages in all_messages:
            for message in messages:
                message["role"] = mapping.get(message["role"], message["role"])
            row["messages"] = messages
            records.append(copy.deepcopy(row))

    random.shuffle(records)

    train_records = []
    val_records = []
    for r in records:
        s = str(r["messages"])
        h = mmh3.hash(s, signed=False)
        if h % 100 < 97:
            train_records.append(r)
        else:
            val_records.append(r)
    with open(train_path, "w") as w:
        for record in train_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
    with open(val_path, "w") as w:
        for record in val_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    fire.Fire(compose_sft_dataset)
