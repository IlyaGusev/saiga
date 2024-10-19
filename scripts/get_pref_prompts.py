import json
import sys
from datasets import load_dataset

input_path = sys.argv[1]
prompts = set()
with open(input_path, "w") as w:
    for row in load_dataset("IlyaGusev/saiga_preferences", split="train"):
        prompt = row["prompt"]
        if str(prompt) in prompts:
            continue
        prompts.add(str(prompt))
        w.write(json.dumps({"messages": prompt}, ensure_ascii=False) + "\n")
