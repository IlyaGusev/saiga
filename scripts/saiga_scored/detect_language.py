from collections import Counter
from typing import Tuple

import fire
from fasttext import load_model as ft_load_model

from src.util.io import read_jsonl, write_jsonl


class FasttextLanguageDetector:
    def __init__(self, model_path: str = "models/lid.176.bin", max_tokens: int = 50) -> None:
        self.model = ft_load_model(model_path)
        self.label_offset = len("__label__")
        self.max_tokens = max_tokens

    def __call__(self, text: str) -> Tuple[str, float]:
        text = text.replace("\xa0", " ").strip()
        text = " ".join(text.split()[: self.max_tokens])

        (label,), (prob,) = self.model.predict(text, k=1)
        label = label[self.label_offset :]
        return label, prob


def detect_language(input_path: str, output_path: str) -> None:
    langid = FasttextLanguageDetector()

    records = []
    for r in read_jsonl(input_path):
        messages = r["messages"]
        messages_str = [m["content"] for m in messages]
        languages = [langid(m)[0] for m in messages_str]
        most_common_language, cnt = Counter(languages).most_common(1)[0]
        if cnt < len(languages) * 2 / 3:
            r["language"] = "Mixed"
            records.append(r)
            continue
        mapping = {"ru": "Russian", "en": "English", "uk": "Ukrainian"}
        if most_common_language not in mapping:
            print(languages)
            continue
        r["language"] = mapping[most_common_language]
        records.append(r)

    write_jsonl(records, output_path)


if __name__ == "__main__":
    fire.Fire(detect_language)
