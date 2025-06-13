import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = sys.argv[1]
output_repo_id = sys.argv[2]
tokenizer = AutoTokenizer.from_pretrained(model_id)
gptq_config = GPTQConfig(bits=8, dataset="c4", tokenizer=tokenizer)
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=gptq_config)
quantized_model.push_to_hub(output_repo_id)
tokenizer.push_to_hub(output_repo_id)
