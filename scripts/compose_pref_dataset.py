import json
import random

import mmh3
import fire
from datasets import load_dataset


def fix_text(text):
     text = text.replace("\xa0", " ").strip()
     text = text.replace("\\xa0", " ").strip()
     return text


def compose_pref_dataset(config_path: str, train_path: str, val_path: str):
    with open(config_path) as r:
        config = json.load(r)

    records = []
    dataset_name = config.get("dataset_name", "IlyaGusev/saiga_preferences")
    revision = config["dataset_revision"]
    field_mapping = config.get("field_mapping", dict())
    for row in load_dataset(dataset_name, split="train", revision=revision):
        if field_mapping:
            for k, v in list(row.items()):
                if k in field_mapping:
                    row[field_mapping[k]] = row.pop(k)

        is_bad_by_regex = row["is_bad_by_regex"]
        if config.get("exclude_regex", False) and is_bad_by_regex:
            continue

        if isinstance(row["chosen"], str):
            row["chosen"] = [{"role": "assistant", "content": row["chosen"]}]
        if isinstance(row["rejected"],  str):
            row["rejected"] = [{"role": "assistant", "content": row["rejected"]}]
        if isinstance(row["prompt"], str):
            row["prompt"] = [{"role": "user", "content": row["prompt"]}]

        max_length_ratio = config.get("max_length_ratio", 2.1)
        max_length_ratio_prob = config.get("max_length_ratio_prob", 0.0)
        if len(str(row["chosen"])) > len(str(row["rejected"])) * max_length_ratio:
            s = str(row["prompt"]) + str(row["chosen"])
            h = mmh3.hash(s, 1337, signed=False)
            if h % 100 > max_length_ratio_prob * 100.0:
                continue

        sonnet_approved_only = config.get("sonnet_approved_only", False)
        if sonnet_approved_only and not row["sonnet_approved"]:
            continue

        if not row["chosen"][0]["content"].strip():
            continue

        if not row["rejected"][0]["content"].strip():
            continue

        row["chosen"][0]["content"] = fix_text(row["chosen"][0]["content"].strip())
        row["rejected"][0]["content"] = fix_text(row["rejected"][0]["content"].strip())

        mapping = {"bot": "assistant"}
        for message in row["prompt"]:
            message["role"] = mapping.get(message["role"], message["role"])
        for message in row["chosen"]:
            message["role"] = mapping.get(message["role"], message["role"])
        for message in row["rejected"]:
            message["role"] = mapping.get(message["role"], message["role"])

        records.append(row)

    random.shuffle(records)

    train_records = []
    val_records = []
    for r in records:
        s = str(r["prompt"])
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
    fire.Fire(compose_pref_dataset)
