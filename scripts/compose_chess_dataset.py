import json
import copy
import random
from itertools import chain

import mmh3
import fire
from datasets import load_dataset
from tqdm import tqdm


def compose_chess_dataset(train_path: str, val_path: str, train_part: int = 98):
    records = []
    dataset_name = "IlyaGusev/lichess_evals_formatted"
    dataset = load_dataset(dataset_name, split="sft")

    hashes = set()
    for row in tqdm(dataset):
        all_messages = []
        if "messages" in row:
            all_messages.append(row["messages"])
        elif "conversation" in row:
            all_messages.append(row["conversation"])
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

            s = str(row["messages"])
            h = mmh3.hash(s, signed=False)
            if h in hashes:
                continue

            hashes.add(h)
            records.append(copy.deepcopy(row))

    random.shuffle(records)

    train_records = []
    val_records = []
    for r in records:
        s = str(r["messages"])
        h = mmh3.hash(s, signed=False)
        if h % 100 < train_part:
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
    fire.Fire(compose_chess_dataset)
