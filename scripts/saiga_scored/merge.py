import json
import random

import fire
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset


def read_jsonl(file_name):
    with open(file_name, encoding="utf-8") as r:
        records = []
        for idx, line in enumerate(r):
            records.append(json.loads(line))
    return records


def generate_ngrams(elements, n: int):
    return {tuple(elements[i:i+n]) for i in range(len(elements) - n + 1)}


def undup_by_ngrams(records, n: int = 8):
    existing_ngrams = dict()
    new_records = []
    records.sort(key=lambda x: x["opus_score"])
    for r in records:
        user_messages = [m for m in r["messages"] if m["role"] == "user"]
        if not user_messages:
            continue
        first_messages = [m["content"] for m in user_messages[:2]]
        words = []
        for m in first_messages:
            words.extend(m.split())
        n_grams = generate_ngrams(words, n)
        skip = False
        for n_gram in n_grams:
            if n_gram in existing_ngrams:
                skip = True
                break
            existing_ngrams[n_gram] = r
        if skip:
            continue
        new_records.append(r)
    print(len(records), len(new_records))
    return new_records


def main(input_path: str, output_path: str) -> None:
    records = read_jsonl(input_path)
    print(len(records))

    clean_records = []
    for row in load_dataset("IlyaGusev/saiga_scored", split="train"):
        clean_records.append(row)
    print(len(clean_records))

    new_records = []
    for r in tqdm(records):
        messages = r["messages"]
        for m in messages:
            if m["role"] == "assistant":
                m["role"] = "bot"
        if "language" not in r:
            r["language"] = "Russian"
        roles = {m["role"] for m in messages}
        if "user" not in roles or "bot" not in roles:
            continue
        r["turns"] = sum([m["role"] == "bot" for m in messages])
        if messages[-1]["role"] != "bot":
            r["messages"] = messages[:-1]

        messages = r["messages"]
        assert messages[-1]["role"] == "bot"
        if not messages[-1]["content"].strip():
            continue
        r["opus_score"] = r.pop("score")
        assert isinstance(r["opus_score"], int)
        r["opus_score"] = max(1, r["opus_score"])
        assert 1 <= r["opus_score"] <= 10, r["opus_score"]
        assert r["turns"] >= 1
        topics = r.pop("topics_answer")
        if not isinstance(topics, dict):
            continue
        r["sonnet_topic"] = topics["topic"]
        r["sonnet_topic_explanation"] = topics["topic_explanation"]
        r["sonnet_complexity"] = topics["complexity"]
        r["sonnet_complexity_explanation"] = topics["complexity_explanation"]
        r.pop("conv_id", None)
        r.pop("orig_messages", None)
        r.pop("char_name", None)
        r.pop("translated_messages", None)
        r.pop("orig_char_name", None)
        new_records.append(r)

    new_records = undup_by_ngrams(new_records)
    print(sum([r["is_bad_by_regex"] is False for r in new_records]))
    print(len(new_records))
    records = clean_records + new_records
    print(len(records))
    random.shuffle(records)
    pd.DataFrame(records).to_parquet(output_path)


if __name__ == "__main__":
    fire.Fire(main)
