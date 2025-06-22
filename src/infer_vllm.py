from typing import Optional
import json
import pathlib

import fire
from vllm import LLM, SamplingParams
from vllm.config import SpeculativeConfig
from transformers import AutoTokenizer
from tqdm import tqdm

from src.util.io import read_jsonl

SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


def infer_vllm(
    model_name: str,
    input_path: str,
    output_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 30,
    max_tokens: int = 2048,
    max_seq_len: int = 8192,
    repetition_penalty: float = 1.1,
    enable_system_prompt: bool = False,
    remove_bos_token: bool = False,
    quantization: Optional[str] = None,
    medusa_path: Optional[str] = None,
    eagle_path: Optional[str] = None,
    max_messages_num: Optional[int] = None,
    max_char_num: Optional[int] = None,
    nrows: Optional[int] = None,
):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )
    speculative_config = None
    if medusa_path:
        cfg = json.load(open(pathlib.Path(medusa_path) / "config.json"))
        num_speculative_tokens = cfg["num_heads"]
        speculative_config = {
            "method": "medusa",
            "model": medusa_path,
            "num_speculative_tokens": num_speculative_tokens,
            "draft_tensor_parallel_size": 1,
        }
    if eagle_path:
        num_speculative_tokens = 3
        speculative_config = {
            "method": "eagle",
            "model": eagle_path,
            "num_speculative_tokens": num_speculative_tokens,
            "draft_tensor_parallel_size": 1,
        }

    llm = LLM(
        model=model_name,
        max_seq_len_to_capture=max_seq_len,
        max_model_len=max_seq_len,
        quantization=quantization,
        speculative_config=speculative_config,
        gpu_memory_utilization=0.8,
        disable_log_stats=False,
    )
    tokenizer = llm.get_tokenizer()
    records = read_jsonl(input_path)
    role_mapping = {
        "bot": "assistant",
        "gpt": "assistant",
        "human": "user",
    }
    prompts = []
    clean_records = []
    for r in tqdm(records, desc="Preprocessing..."):
        if "instruction" in r:
            messages = [{"role": "user", "content": r["instruction"]}]
        elif "messages" in r or "prompt" in r:
            messages = r.get("messages", r.get("prompt"))

        assert messages
        if messages[0]["role"] != "system" and enable_system_prompt:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        for m in messages:
            m["role"] = role_mapping.get(m["role"], m["role"])
        if max_messages_num is not None:
            messages = messages[:max_messages_num]
        if messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        r["messages"] = messages

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if remove_bos_token:
            prompt = prompt.replace(tokenizer.bos_token, "")
        if max_char_num is not None and len(prompt) >= max_char_num:
            continue
        prompts.append(prompt)
        clean_records.append(r)
        if nrows and len(clean_records) >= nrows:
            break

    print(prompts[0])
    outputs = llm.generate(prompts, sampling_params)
    with open(output_path, "w") as w:
        for record, output in zip(clean_records, outputs):
            prompt = output.prompt
            prompt_token_ids = output.prompt_token_ids
            assert prompt_token_ids[0] != prompt_token_ids[1], prompt_token_ids
            generated_text = output.outputs[0].text
            generated_text = generated_text.encode("utf-8").decode("utf-8", "ignore")
            print(prompt)
            print(generated_text)
            print(prompt_token_ids)
            print()
            print()
            record["messages"].append({"role": "assistant", "content": generated_text})
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")

    if not eagle_path and not medusa_path:
        return

    metrics = llm.get_metrics()
    num_drafts = num_accepted = 0
    acceptance_counts = [0] * num_speculative_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            num_accepted += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]
    print("-" * 50)
    print(f"mean acceptance length: {1 + (num_accepted / num_drafts):.2f}")
    print("-" * 50)

    for i in range(len(acceptance_counts)):
        print(f"acceptance at token {i}:{acceptance_counts[i] / num_drafts:.2f}")


if __name__ == "__main__":
    fire.Fire(infer_vllm)
